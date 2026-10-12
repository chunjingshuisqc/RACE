$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$Root\RACEVoteCore\run_paired.py" @args
