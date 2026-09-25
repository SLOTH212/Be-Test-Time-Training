import sys
sys.path.insert(0,'/path/to/ttt/work/llama31_8b_stage1_stage2_pipeline_formal_v1')
from pipeline import main
sys.argv.insert(1,"launch" if "--execute" in sys.argv else "verify")
main()
