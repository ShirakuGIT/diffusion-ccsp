"""Run Diffusion-CCSP baseline evaluation using their own pipeline."""
from train_utils import load_trainer

RUN_ID    = 'qsd3ju74'
MILESTONE = 7

TEST_TASKS = {i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split"
              for i in range(2, 6)}


def main():
    trainer = load_trainer(RUN_ID, MILESTONE, verbose=False,
                           input_mode='qualitative', test_tasks=TEST_TASKS)
    print(f'Loaded Diffusion-CCSP (run={RUN_ID}, milestone={MILESTONE})')
    print(f'test_tasks={list(trainer.test_dls.keys())}')

    trainer.evaluate('diffusion_eval', tries=(10, 0), verbose=True, save_log=True)


if __name__ == '__main__':
    main()
