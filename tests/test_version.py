# -*- coding: utf-8 -*-
"""version.json 的格式，以及 scripts/bump_version.py 的升级规则（末位 +1，不进位）。

运行：python -m unittest discover -s tests"""
import datetime
import importlib.util
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bump_version", os.path.join(ROOT, "scripts", "bump_version.py"))
bv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv)


class Version(unittest.TestCase):
    def test_file_format(self):
        d = json.load(open(os.path.join(ROOT, "version.json"), encoding="utf-8"))
        self.assertRegex(d["version"], r"^\d+\.\d+\.\d+$")
        datetime.date.fromisoformat(d["updated_at"])
        self.assertIn("engines", d)

    def test_bump_last_digit(self):
        self.assertEqual(bv.bump("0.1.2"), "0.1.3")
        self.assertEqual(bv.bump("0.1.9"), "0.1.10")
        self.assertEqual(bv.bump("1.0.0"), "1.0.1")
        with self.assertRaises(ValueError):
            bv.bump("0.1")


if __name__ == "__main__":
    unittest.main()
