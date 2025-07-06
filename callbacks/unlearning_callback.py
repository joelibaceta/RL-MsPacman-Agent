from stable_baselines3.common.callbacks import BaseCallback

class TemporaryUnlearningCallback(BaseCallback):
    """
    Callback que, al cumplirse una condición (episodio largo o t > umbral),
    fuerza ε = high_eps durante hold_steps pasos, y luego retoma la schedule original.
    """
    def __init__(
        self,
        len_threshold: float = 90,
        step_threshold: int = 5_200_000,
        high_eps: float = 1.0,
        hold_steps: int = 1_000_000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.len_threshold  = len_threshold
        self.step_threshold = step_threshold
        self.high_eps       = high_eps
        self.hold_steps     = hold_steps

        # Señales internas
        self.triggered      = False
        self.trigger_step   = None
        self.orig_schedule  = None

    def _on_rollout_end(self) -> None:
        if self.triggered:
            return

        # Chequeo de condiciones
        ep_len_mean = self.logger.name_to_value.get('rollout/ep_len_mean', 0)
        if self.num_timesteps > self.step_threshold:
            reason = f"timestep {self.num_timesteps} > {self.step_threshold}"
        elif ep_len_mean > self.len_threshold:
            reason = f"ep_len_mean {ep_len_mean:.1f} > {self.len_threshold}"
        else:
            return

        # Guardamos la schedule original y disparamos el hold
        self.orig_schedule = self.model.exploration_schedule
        self.triggered     = True
        self.trigger_step  = self.num_timesteps

        if self.verbose:
            print(f"\n⚠️  Unlearning ON at step {self.num_timesteps} ({reason}), "
                  f"holding ε={self.high_eps} for {self.hold_steps} steps.")

    def _on_step(self) -> bool:
        if not self.triggered:
            return True

        t0 = self.trigger_step
        t  = self.num_timesteps

        if t <= t0 + self.hold_steps:
            # Durante el hold, forzamos ε = high_eps
            self.model.exploration_rate = self.high_eps
        else:
            # Pasado el hold, restauramos schedule original y salimos
            if self.verbose:
                print(f"🔄  Hold terminado en step {t}, "
                      f"restaurando decay original.")
            # Restauramos la schedule para que retome el decay normal
            self.model.exploration_schedule = self.orig_schedule
            # Fijamos exploration_rate de acuerdo con la schedule restaurada
            # (recalculamos el progress en [0,1])
            progress = min(1.0, t / self.model._total_timesteps)
            self.model.exploration_rate = self.model.exploration_schedule(progress)

            # Desactivamos el callback para no volver a entrar
            self.triggered = False

        return True