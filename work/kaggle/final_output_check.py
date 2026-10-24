"""Verify the pinned compact benchmark and publish exactly the FIRST test answer."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from final_vote import validate_answers
from final_session import INPUT_SHA


def check(directory, test_file):
    directory, test_file = Path(directory), Path(test_file)
    assert hashlib.sha256(test_file.read_bytes()).hexdigest() == INPUT_SHA['test.json']
    tests = json.loads(test_file.read_text())
    assert len(tests) == 1919
    order_digest = hashlib.sha256(json.dumps([r['question'] for r in tests], ensure_ascii=False).encode()).hexdigest()
    report = json.loads((directory/'summary.json').read_text())
    digests, timings = [], []
    for i in range(1, 3):
        path = directory/f'test-{i}'
        answers = path/'answers.jsonl'
        validate_answers(answers, 1919)
        config = json.loads((path/'config.json').read_text())
        assert 'reference' not in config and config['expected_rows'] == 1919
        timing = json.loads((path/'summary.json').read_text())
        assert timing['status'] == 'complete' and timing['rows'] == 1919
        assert timing['full_pipeline_seconds'] <= 120. and timing['full_pipeline_seconds'] < 180.
        assert timing['deadline_guard']['fallback_count'] == 0
        assert not timing['deadline_guard']['triggered']
        assert timing['winner_reexecution']['complete'] and not timing['winner_reexecution']['mismatch_rows']
        assert timing['single_gpu'] is True and timing['question_order_sha256'] == order_digest
        assert timing['input_sha256'] == INPUT_SHA['test.json']
        digest = hashlib.sha256(answers.read_bytes()).hexdigest()
        assert digest == timing['answers_sha256']
        assert abs(sum(timing['phases_seconds'].values())-timing['full_pipeline_seconds']) < .01
        digests.append(digest); timings.append(timing)
    assert len(set(digests)) == 1
    assert (directory/'answers.jsonl').read_bytes() == (directory/'test-1/answers.jsonl').read_bytes()
    assert report['GPU_runs'] == 3 and report['session_seconds'] <= 900.
    assert report['status'] == 'passed' and all(report['gates'].values())
    quality = json.loads((directory/'tune/quality.json').read_text())
    validate_answers(directory/'tune/answers.jsonl', 2048)
    assert quality['n'] == 2048 and quality['passed'] and abs(quality['delta_pp']) <= 1.
    result = {'passed': True, 'test': timings, 'tune': quality, 'answers_sha256': digests[0],
              'question_order_sha256': order_digest, 'format_and_order_checked': True,
              'deterministic': True, 'fallback_count': 0, 'source': 'test-1'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--test', type=Path, default=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/test.json')
    args = parser.parse_args()
    result = check(args.directory, args.test)
    destination = WORK/'submissions/final-mixed-k4k4-l2-28800.jsonl'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        assert destination.read_bytes() == (args.directory/'test-1/answers.jsonl').read_bytes(), 'Do not overwrite a different submission'
    else:
        shutil.copyfile(args.directory/'test-1/answers.jsonl', destination)
    validate_answers(destination, 1919)
    (WORK/'checks/final-inference-output.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': True, 'submission': str(destination), 'sha256': result['answers_sha256']}))
