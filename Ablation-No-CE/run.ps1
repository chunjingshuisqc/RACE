$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$Here\..\RACEVoteCore\run_experiment.py" --config "$Here\config.json" @args
