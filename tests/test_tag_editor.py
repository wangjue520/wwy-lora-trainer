# -*- coding: utf-8 -*-
"""标签编辑后端（app/captionutil.py 的拆分 / 编码、app/dataset.py 的批量操作 / 撤销、app/taglint.py 的体检）。

运行：python -m unittest discover -s tests"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import captionutil as cu  # noqa: E402
import dataset  # noqa: E402
import paths  # noqa: E402
import taglint  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.d = os.path.join(self.tmp, "ds")
        os.makedirs(self.d)
        self.patches = [mock.patch.object(paths, "BACKUPS", os.path.join(self.tmp, "bk")),
                        mock.patch.object(dataset, "_trigger_main", side_effect=lambda d: self.trigger)]
        self.trigger = None
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def mk(self, name, cap=None, raw=None):
        img = os.path.join(self.d, name + ".png")
        open(img, "wb").write(b"x")
        if raw is not None:
            open(img[:-4] + ".txt", "wb").write(raw)
        elif cap is not None:
            open(img[:-4] + ".txt", "w", encoding="utf-8").write(cap)
        return img

    def cap(self, name):
        return open(os.path.join(self.d, name + ".txt"), encoding="utf-8").read().strip()


class Parse(unittest.TestCase):
    def test_mixed(self):
        tags, text, sep = cu.parse_caption("k2, 1girl, mosaic censoring. A girl is standing, looking at viewer. She smiles.\n")
        self.assertEqual(tags, ["k2", "1girl", "mosaic censoring"])
        self.assertEqual(text, "A girl is standing, looking at viewer. She smiles.")
        self.assertEqual(cu.build_caption(tags, text, sep), "k2, 1girl, mosaic censoring. A girl is standing, looking at viewer. She smiles.")

    def test_trigger_then_sentence(self):
        tags, text, sep = cu.parse_caption("k2, A cat is sitting on a bench, looking up.")
        self.assertEqual((tags, sep), (["k2"], ", "))
        self.assertTrue(text.startswith("A cat"))

    def test_symbol_tags_and_lowercase_dot(self):
        self.assertEqual(cu.parse_caption("..., ^_^, dr. stone")[0], ["...", "^_^", "dr. stone"])

    def test_prose_only(self):
        self.assertEqual(cu.parse_caption("A girl with long hair, standing in a field of flowers.")[0], [])
        self.assertEqual(cu.parse_caption("一个女孩站在花丛中，看着镜头，阳光很好，背景是蓝天白云。")[0], [])

    def test_norm(self):
        self.assertEqual(cu.norm_tag("Mosaic_Censoring"), "mosaic censoring")
        self.assertEqual(cu.norm_tag("\ufeffkafka \\(honkai\\)"), "kafka (honkai)")
        self.assertEqual(cu.norm_tag("^_^"), "^_^")
        self.assertEqual(cu.split_input("censored，mosaic censoring,"), ["censored", "mosaic censoring"])


class Encoding(Base):
    def test_bom_stripped(self):
        img = self.mk("a", raw="censored, 1girl".encode("utf-8-sig"))
        self.assertEqual(cu.read_caption_ex(img), ("censored, 1girl", "bom"))
        r = dataset.batch(self.d, "remove", None, ".txt", tags="censored")
        self.assertEqual(r["changed"], 1)
        self.assertEqual(open(img[:-4] + ".txt", "rb").read(), b"1girl\n")

    def test_gbk_not_touched(self):
        img = self.mk("a", raw="1girl, 白丝".encode("gbk"))
        r = dataset.batch(self.d, "add", None, ".txt", tags="smile")
        self.assertEqual((r["changed"], r["skipped"]), (0, [img]))
        self.assertEqual(open(img[:-4] + ".txt", "rb").read(), "1girl, 白丝".encode("gbk"))
        with self.assertRaises(ValueError):
            dataset.save_one(self.d, img, "x")
        r = dataset.convert_encoding(self.d, [img], "gb18030")
        self.assertEqual(self.cap("a"), "1girl, 白丝")
        self.assertEqual(r["changed"], 1)


class Batch(Base):
    def test_remove_variants_and_real_count(self):
        self.mk("a", "1girl, solo, censored, mosaic censoring, long hair\n")
        self.mk("b", "1girl, mosaic_censoring, Censored\n")
        self.mk("c", "1girl, smile\n")
        r = dataset.batch(self.d, "remove", None, ".txt", tags="censored,mosaic censoring,")
        self.assertEqual((r["changed"], r["total"]), (2, 3))
        self.assertEqual(self.cap("a"), "1girl, solo, long hair")
        self.assertEqual(self.cap("b"), "1girl")
        r = dataset.batch(self.d, "remove", None, ".txt", tags="censored")
        self.assertEqual(r["changed"], 0)
        self.assertNotIn("undo_id", r)   # 没改东西不留备份
        self.assertEqual(len(dataset.history(self.d)), 1)

    def test_add_front_moves_existing_after_trigger(self):
        self.trigger = {"main": "k2"}
        self.mk("a", "k2, 1girl, censored\n")
        self.mk("b", "k2, censored, 1girl\n")
        self.mk("c", "1girl\n")
        r = dataset.batch(self.d, "add", None, ".txt", tags="censored", position="front")
        self.assertEqual(r["changed"], 2)
        self.assertEqual(self.cap("a"), "k2, censored, 1girl")
        self.assertEqual(self.cap("b"), "k2, censored, 1girl")
        self.assertEqual(self.cap("c"), "censored, 1girl")

    def test_add_front_guesses_trigger_without_record(self):
        for n in "abc":
            self.mk(n, "k2, 1girl, smile\n")
        dataset.batch(self.d, "add", None, ".txt", tags="censored", position="front")
        self.assertEqual(self.cap("a"), "k2, censored, 1girl, smile")

    def test_add_back_keeps_sentence(self):
        self.mk("a", "k2, 1girl, solo. A girl is standing, looking at viewer. She smiles.\n")
        dataset.batch(self.d, "add", None, ".txt", tags="blue sky", position="back")
        self.assertEqual(self.cap("a"), "k2, 1girl, solo, blue sky. A girl is standing, looking at viewer. She smiles.")
        dataset.batch(self.d, "add", None, ".txt", tags="Blue_Sky", position="back")
        self.assertEqual(self.cap("a").count("blue sky"), 1)

    def test_empty_scope_is_not_everything(self):
        self.mk("a", "1girl\n")
        r = dataset.batch(self.d, "clear", [], ".txt")
        self.assertEqual(r["changed"], 0)
        self.assertEqual(self.cap("a"), "1girl")

    def test_rename_map_dedupes(self):
        self.mk("a", "long_hair, Long Hair, smile\n")
        dataset.batch(self.d, "rename_map", None, ".txt", map={"long_hair": "long hair"})
        self.assertEqual(self.cap("a"), "long hair, smile")

    def test_rename_leaves_other_dupes_and_files_alone(self):
        self.mk("a", "1girls, 1girl, smile, smile\n")
        self.mk("b", "solo, solo\n")
        r = dataset.batch(self.d, "rename_map", None, ".txt", map={"1girls": "1girl"})
        self.assertEqual(r["changed"], 1)
        self.assertEqual(self.cap("a"), "1girl, smile, smile")
        self.assertEqual(self.cap("b"), "solo, solo")

    def test_bad_regex_message(self):
        self.mk("a", "1girl\n")
        with self.assertRaises(ValueError):
            dataset.batch(self.d, "find_replace", None, ".txt", find="(", replace="", regex=True)


class Undo(Base):
    def test_cascade_and_modified_check(self):
        a = self.mk("a", "1girl, smile\n")
        r1 = dataset.batch(self.d, "add", None, ".txt", tags="x")
        dataset.batch(self.d, "add", None, ".txt", tags="y")
        self.assertEqual(self.cap("a"), "1girl, smile, x, y")
        dataset.save_one(self.d, a, "1girl, smile, x, y, manual")
        chk = dataset.undo(self.d, ".txt", r1["undo_id"], dry=True)
        self.assertEqual((chk["modified"], chk["count"]), (1, 2))
        r = dataset.undo(self.d, ".txt", r1["undo_id"])
        self.assertTrue(r["ok"])
        self.assertEqual(self.cap("a"), "1girl, smile")
        self.assertEqual(dataset.history(self.d), [])

    def test_save_conflict(self):
        a = self.mk("a", "1girl\n")
        self.assertTrue(dataset.save_one(self.d, a, "1girl, smile", base="1girl\n")["ok"])
        open(a[:-4] + ".txt", "w", encoding="utf-8").write("1girl, from tagger\n")
        r = dataset.save_one(self.d, a, "1girl, smile, blush", base="1girl, smile")
        self.assertTrue(r["conflict"])
        self.assertEqual(self.cap("a"), "1girl, from tagger")

    def test_remove_image_same_name(self):
        os.makedirs(os.path.join(self.d, "sub"))
        a = self.mk("a", "1girl")
        b = os.path.join(self.d, "sub", "a.png")
        open(b, "wb").write(b"y")
        open(b[:-4] + ".txt", "w").write("2girls")
        dataset.remove_image(self.d, a)
        dataset.remove_image(self.d, b)
        moved = sorted(os.listdir(os.path.join(self.d, "_已移除")))
        self.assertEqual(moved, ["a.png", "a.txt", "a_2.png", "a_2.txt"])


class Lint(Base):
    def test_issues(self):
        self.trigger = {"main": "k2"}
        self.mk("a", "k2, 1girl, long_hair, solo\n")
        self.mk("b", "1girl, long hair, Long Hair\n")
        self.mk("c", "k2, 1girl, 2girls\n")
        self.mk("d", raw="k2, 1girl".encode("utf-8-sig"))
        self.mk("e")
        with mock.patch.dict(sys.modules, {"tagdict": mock.Mock(load=lambda: False)}):
            r = taglint.run(self.d, ".txt", limit=225)
        codes = {i["code"]: i for i in r["issues"]}
        self.assertEqual(set(codes), {"variants", "dupes", "conflict", "trigger", "bom", "empty", "nodict"})
        self.assertEqual(len(codes["trigger"]["files"]), 1)
        fx = codes["variants"]["fixes"][0]
        taglint.fix(self.d, fx["id"], fx["data"])
        self.assertEqual(self.cap("a"), "k2, 1girl, long hair, solo")
        self.assertEqual(self.cap("b"), "1girl, long hair")

    def test_dict_alias_and_unknown(self):
        self.mk("a", "1girl, hatsune_miku_(cosplay), lnog hair, my_trigger_word\n")
        self.mk("b", "1girl, long hair, blue eyes, short hair\n")
        by = {"1girl": 1, "long_hair": 1, "blue_eyes": 1, "short_hair": 1}
        fake = mock.Mock(load=lambda: True, _by_tag=by,
                         resolve=lambda q: q if q in by else ("long_hair" if q == "longhair" else ("1girl" if q == "hatsune_miku_(cosplay)" else None)),
                         search=lambda q, n: [{"name": "long_hair"}] if "lnog" in q else [], fuzzy=lambda q, n: [])
        self.trigger = {"main": "my_trigger_word"}
        with mock.patch.dict(sys.modules, {"tagdict": fake}):
            r = taglint.run(self.d, ".txt")
        codes = {i["code"]: i for i in r["issues"]}
        self.assertEqual(codes["alias"]["rows"][0]["to"], "1girl")
        unk = codes["unknown"]["rows"]
        self.assertEqual([x["tag"] for x in unk], ["lnog hair"])   # 触发词不算「字典里没有」
        self.assertEqual(unk[0]["suggest"], ["long hair"])          # 建议按训练集的写法习惯（空格）


if __name__ == "__main__":
    unittest.main()
