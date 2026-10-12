$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$Here\..\RACEVoteCore\analyze_experiment.py" @args --config "$Here\config.json"
