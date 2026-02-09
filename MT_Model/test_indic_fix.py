# test_indic_fix.py
import sys
sys.path.append(".")  # ensure current dir is in path

from mt_model import translate_with_indictrans2, ISO_TO_FLORES

def run_tests():
    # English → Hindi
    text_en = "Hello, how are you?"
    src, tgt = ISO_TO_FLORES["en"], ISO_TO_FLORES["hi"]
    print("\n>> en → hi (IndicTrans2)")
    print("Input :", text_en)
    print("Output:", translate_with_indictrans2(text_en, src, tgt))

    # Hindi → English
    text_hi = "भारत एक महान देश है।"
    src, tgt = ISO_TO_FLORES["hi"], ISO_TO_FLORES["en"]
    print("\n>> hi → en (IndicTrans2)")
    print("Input :", text_hi)
    print("Output:", translate_with_indictrans2(text_hi, src, tgt))

    # Hindi → Marathi (Indic ↔ Indic)
    text_hi2 = "मैं कल स्कूल जाऊँगा।"
    src, tgt = ISO_TO_FLORES["hi"], ISO_TO_FLORES["mr"]
    print("\n>> hi → mr (IndicTrans2)")
    print("Input :", text_hi2)
    print("Output:", translate_with_indictrans2(text_hi2, src, tgt))


if __name__ == "__main__":
    run_tests()
