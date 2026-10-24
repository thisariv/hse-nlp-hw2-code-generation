# Append this cell to the already completed 1000-step notebook in the SAME runtime.
# Do not rerun setup, model initialization, or the original training cell.
TOTAL_STEPS = 3000

required = ['model', 'optimizer', 'criterion', 'result', 'config', 'small_train',
            'training_log', 'generation_log', 'training_batch', 'evaluate_gold',
            'evaluate_generation', 'small_train_hash', 'BATCH_SIZE', 'VOCAB',
            'GENERATION_CAP', 'GENERATION_EVERY', 'torch', 'nn', 'random',
            'hashlib', 'json', 'time', 'Path']
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError('Нужна та же Colab-сессия после 1000 шагов. '
                       'Скопируй эту ячейку в конец выполненного notebook. '
                       'Если runtime сброшен, нужны checkpoint и JSON; '
                       'outputs notebook сами по себе не содержат веса.')


def continue_overfit16():
    start_step = result['actual_steps']
    assert 1000 <= start_step <= TOTAL_STEPS, 'Ожидаются уже выполненные 1000 шагов'
    expected_hash = 'd09cd3121ebfaac0707b52136008a00f6389fed0b3eec3b4a1c502e739bdeb6b'
    actual_hash = hashlib.sha256(json.dumps(small_train, ensure_ascii=False,
                                            sort_keys=True).encode()).hexdigest()
    assert actual_hash == small_train_hash == result['small_train_sha256'] == expected_hash
    expected_config = {'seed': 42, 'batch_size_train': 8, 'd_model': 128,
        'encoder_layers': 2, 'decoder_layers': 2, 'heads': 4, 'ffn_dim': 512,
        'dropout': 0.1, 'norm_first': False, 'lr': 0.001, 'weight_decay': 0.01,
        'clip_norm': 1.0, 'scheduler': None, 'tokenizer': 'UTF-8 bytes',
        'max_positions': 2048, 'generation_cap': 512}
    assert all(config.get(k) == v for k, v in expected_config.items()), 'Настройки изменились'
    assert result['config'] is config and result['training_log'] is training_log
    assert result['generation_log'] is generation_log
    assert len(small_train) == 16 and BATCH_SIZE == 8 and VOCAB == 259
    assert GENERATION_CAP == 512 and GENERATION_EVERY == 250
    assert training_log[-1]['step'] == start_step
    assert next(model.parameters()).dtype == torch.float32
    assert sum(p.numel() for p in model.parameters()) == 1254915
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters()}
    assert all(group['lr'] == 0.001 and group['weight_decay'] == 0.01
               for group in optimizer.param_groups)
    assert {int(state['step'].item()) for state in optimizer.state.values()} == {start_step}, \
        'Счётчик optimizer не совпадает с результатами; обучение не начинаем заново'

    results_file = Path('hw02-overfit-16-3000-results.json')
    checkpoint_file = 'hw02-overfit-16-3000.pt'
    if result['memorization_passed'] or start_step == TOTAL_STEPS:
        print('Уже завершено: дальнейшие обновления не нужны.', flush=True)
        return

    previous_seconds = result['wall_seconds_including_diagnostics_and_checkpointing']
    torch.cuda.synchronize()
    continued_started = time.perf_counter()
    config['max_steps'] = TOTAL_STEPS
    result.setdefault('continuation_from_step', start_step)
    result['run_finished'] = False
    print('CONTINUE', json.dumps({'from_step': start_step, 'total_steps_limit': TOTAL_STEPS,
                                  'maximum_extra_updates': TOTAL_STEPS - start_step}), flush=True)

    def save_progress(step):
        torch.save({'model': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'steps': step, 'config': config,
            'small_train_sha256': small_train_hash,
            'python_random_state': random.getstate(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all()}, checkpoint_file)
        result['actual_steps'] = step
        result['wall_seconds_including_diagnostics_and_checkpointing'] = (
            previous_seconds + time.perf_counter() - continued_started)
        results_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    for step in range(start_step + 1, TOTAL_STEPS + 1):
        model.train()
        src, tgt, labels = training_batch(random.sample(small_train, BATCH_SIZE))
        logits = model(src, tgt)
        loss = criterion(logits.reshape(-1, VOCAB), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == TOTAL_STEPS:
            entry = {'step': step, 'train_batch_loss': loss.item(), **evaluate_gold()}
            training_log.append(entry)
            print('LOSS', json.dumps(entry), flush=True)
            if step % GENERATION_EVERY == 0 or step == TOTAL_STEPS:
                checked = {'step': step, **evaluate_generation()}
                generation_log.append(checked)
                result['memorization_passed'] = checked['summary']['exact_tokens_with_eos_count'] == 16
                print('GREEDY', json.dumps({'step': step, **checked['summary']}), flush=True)
            save_progress(step)
            if result['memorization_passed']:
                print('Все 16 программ воспроизведены точно с EOS; останавливаемся.', flush=True)
                break

    torch.cuda.synchronize()
    result['run_finished'] = True
    result['wall_seconds_including_diagnostics_and_checkpointing'] = (
        previous_seconds + time.perf_counter() - continued_started)
    results_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    last_generation = generation_log[-1]
    examples = last_generation['examples']
    failures = [e for e in examples if not e['exact_tokens_with_eos']]
    for example in (failures or examples)[:3]:
        print('EXAMPLE', json.dumps(example, ensure_ascii=False), flush=True)
    print('RESULT', json.dumps({'actual_steps': result['actual_steps'],
        'max_steps': TOTAL_STEPS, 'memorization_passed': result['memorization_passed'],
        'last_teacher_forcing': training_log[-1], 'last_greedy': last_generation['summary'],
        'wall_seconds_including_diagnostics_and_checkpointing':
            result['wall_seconds_including_diagnostics_and_checkpointing']}), flush=True)
    print('Saved:', str(results_file), checkpoint_file)
    print('Сохрани notebook с outputs и скачай новый JSON и checkpoint.')


continue_overfit16()
