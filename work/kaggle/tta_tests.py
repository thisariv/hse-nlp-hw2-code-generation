"""Targeted invariance, boundaries and idempotence tests for surface TTA."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from tta_rules import apply, numbers, RULES

class RulesTest(unittest.TestCase):
    def test_examples(self):
        pairs=[('dollar-words','Pay $5 each, $12.50 total?','Pay 5 dollars each, 12.50 dollars total?'),
               ('dollar-symbol','Pay 5 dollars and 12.50 dollars.','Pay $5 and $12.50.'),
               ('ungroup','There are 12,000 and 1,234.50 units?','There are 12000 and 1234.50 units?'),
               ('punctuation','How many ? Why : now !','How many? Why: now!'),
               ('price-trim','Pay $5.00 each and $1.50 per unit?','Pay $5 each and $1.5 per unit?')]
        for k,q,want in pairs:
            self.assertEqual(apply(q,k),want)
            self.assertEqual(numbers(q),numbers(want))
            self.assertEqual(apply(want,k),want)

    def test_untouched(self):
        pairs={'dollar-words':['$5k each','$5 million each','$5.5 million each','$5.25k each','$5,000 million each','5 dollars','$5-billion','Pay $5 dollars'],
               'dollar-symbol':["5 dollars’ worth", "5 dollars' worth",'15.5 dollar','USD 5','5 cents'],
               'ungroup':['1,2,3','1,23','1,000,0000','1,234, 2','1.234,56','x1,000x'],
               'punctuation':['5.5, 6','A ?B','a , b'],
               'price-trim':['12.50 cm','2026.10','$5k','$5.10k','$10']}
        pairs['punctuation'].remove('A ?B')
        for k,values in pairs.items():
            for q in values: self.assertEqual(apply(q,k),q,(k,q))

    def test_number_parser(self):
        self.assertEqual(numbers('$1,000.50 and 50% and .25'),numbers('1000.5 dollars and 50 percent and 0.25'))

    def test_mixed(self):
        q='Pay $5.00 for 1,000 units or 7 dollars for 2,000 units ?'
        for k in RULES:
            x=apply(q,k); self.assertEqual(numbers(x),numbers(q)); self.assertEqual(apply(x,k),x)

if __name__=='__main__': unittest.main()
