"""Run the PPO+NGOpt schedule with a frozen local LLM replacing NGOpt."""

from ppo_ngopt_rlgo.gym_train import main


if __name__ == "__main__":
    main(method="ppo_llm")
