# PowerShell script for Nsight Compute profiling

# Define parameters
$metrics = "smsp__inst_executed.sum,smsp__inst_executed.sum.per_second,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fmad_pred_on.sum,smsp__sass_thread_inst_executed_op_iadd_pred_on.sum,smsp__sass_thread_inst_executed_op_imad_pred_on.sum,smsp__sass_thread_inst_executed_op_imul_pred_on.sum,smsp__sass_thread_inst_executed_op_idiv_pred_on.sum,smsp__sass_thread_inst_executed_op_shl_pred_on.sum,smsp__sass_thread_inst_executed_op_shr_pred_on.sum,smsp__sass_thread_inst_executed_op_atom_pred_on.sum,smsp__sass_thread_inst_executed_op_cvt_pred_on.sum,smsp__sass_thread_inst_executed_op_bit_pred_on.sum"
$target_processes = "all"
$set = "full"
$output_dir = "latest"
$exe_path = ".\out\build\x64-Debug\src\gpu\gpu.exe"
$input_file = ".\dataset\video.avi"
$output_file = ".\dataset\output.mp4"

try {
    # Get all available sections
    Write-Host "Getting available sections..."
    $sections_list = ncu --list-sections | Out-String
    $sections_list = $sections_list -split "`n" | select -Skip 3 # Skip the first 3 lines
    $sections_list = $sections_list -split "`r" | select -Skip 3 # Skip the first 3 lines (for Windows style line endings)

    # Extract section names and join them into a comma separated string
    $sections = @()
    foreach($line in $sections_list) {
        if ($line -eq "") { break } # Stop at the first empty line
        $section = $line -split "\s+" | select -Index 0
        $sections += $section
    }
    $sections = $sections -join " --section "

    Write-Host "Available sections: $sections"

    # Prepare and perform profiling command
    $command = "ncu --metrics $metrics --target-processes $target_processes --set $set -o $output_dir -f $exe_path -i $input_file -o $output_file"
    Write-Host "Starting profiling with Nsight Compute..."
    Write-Host "Command: $command"
    Invoke-Expression $command

    Write-Host "Profiling completed successfully."
}
catch {
    Write-Host "An error occurred during the profiling process:"
    Write-Host $_.Exception.Message
}
