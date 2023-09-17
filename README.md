# Masterarbeit Timo Langner 

Repository zur Datensicherung der Masterarbeit "Künstliche Intelligenz in FPGAs" 

##HW 
- Pre-Processing Kernel
- Post-Processing Kernel
- Xilinx Deep Learning Processing Unit (B4096)
- Tracking (Kalman-Filter Kernel + Ungarische Methode Kernel)

##SW 
- Software zum Steuern der Kernel über Xilinx Runtime Library und Vitis AI Runtime Library
- Bash-Skript zum Kompilieren des Programms
- Bibliotheken für Vitis AI --> auf MPSOC installieren

##Device Binary 
- Kernel Objektdateien
- Konfigurationsdateien für Packaging und Kernel-Linking
- Base-Plattform
- Common-Image / Custom-Image

##How-To

1. SD-Card Image erstellen mit V++ Kernel-Linking und Packaging
2. src/mpsoc auf FPGA installieren 
3. Programm mit app_build.sh kompilieren
4. src/bin auf FPGA kopieren
5. Programm ausführen:
   ./tracking.exe <Kamera(0)/Dateipfad(1)> <HW-Processing(0)/SW-Processing(1)> Yolo_Tracking.xmodel dpu.xclbin <Ordner mit Bildern> <ROI Bild im 416x416 Pixel Format>
