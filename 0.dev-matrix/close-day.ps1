param(
  [int]$TopTasks = 3
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$WorkspaceRoot = Split-Path $RepoRoot -Parent
$SharedScript = Join-Path $WorkspaceRoot '0.dev-matrix\close-day.repo.shared.ps1'

if (-not (Test-Path $SharedScript)) {
  Write-Host 'Shared close-day helper not found.' -ForegroundColor Yellow
  return
}

& $SharedScript -RepoRoot $RepoRoot -TopTasks $TopTasks
