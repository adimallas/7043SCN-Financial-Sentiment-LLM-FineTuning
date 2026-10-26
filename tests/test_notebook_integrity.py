"""Static reproducibility checks; no network access or Colab GPU is required."""
import ast
from pathlib import Path
import unittest
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'Sampath_Adimalla_17522555_7043SCN_Task1_Financial_Sentiment_FineTuning.ipynb'

class NotebookIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(cls.nb)
        cls.code='\n'.join(c.source for c in cls.nb.cells if c.cell_type=='code')

    def test_code_cells_parse(self):
        for cell in self.nb.cells:
            if cell.cell_type=='code' and not cell.source.lstrip().startswith('%'):
                ast.parse(cell.source)

    def test_baseline_and_adapters(self):
        for must in ('AutoModelForSeq2SeqLM','get_peft_model',
                     'PeftModel.from_pretrained','train_variant','baseline',
                     'full_finetuning','lora_r','Seq2SeqTrainer'):
            with self.subTest(must=must): self.assertIn(must,self.code)

    def test_locked_test_comes_after_training(self):
        self.assertLess(self.code.index("train_variant('full_finetuning'"),
                        self.code.index("for tag in FINAL_TAGS:"))

    def test_splitting_leakage_validation(self):
        for must in ('stratify=df.label','sets[0]&sets[1]','adapter_selection.json',
                     'best_model_checkpoint', 'source_sha256'):
            with self.subTest(must=must): self.assertIn(must,self.code)

    def test_statistical_exports(self):
        for must in ('bootstrap_f1','paired_bootstrap_differences.csv',
                     'macro_f1_bootstrap_ci.csv','confusion_matrix',
                     'invalid_outputs', 'MODEL_CARD.md'):
            with self.subTest(must=must): self.assertIn(must,self.code)

if __name__=='__main__':
    unittest.main()
