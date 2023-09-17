Linken der Kernel Objekte

v++ -t hw --platform ~/platform/xilinx_zcu102_base_202210_1/xilinx_zcu102_base_202210_1.xpfm --link ../preprocess.xo ../dpu.xo ../postprocess.xo ../kalman.xo ../alignmem.xo ../hungarian.xo -o'dpu.xclbin' --config ../linker_config.cfg --save-temps --optimize 2 --vivado.impl.strategies "Performance_NetDelay_low,Performance_NetDelay_high"

Packaging 

v++ --package -t hw --platform ~/platform/xilinx_zcu102_base_202210_1/xilinx_zcu102_base_202210_1.xpfm --save-temps ./dpu.xclbin --config ../config.cfg

