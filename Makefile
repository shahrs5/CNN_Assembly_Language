GCC_PREFIX = riscv32-unknown-elf
ABI = -march=rv32gcv -mabi=ilp32d
LINK = link.ld

all: compile execute parse

clean:
	rm -f *.txt *.hex *.dis *.exe

compile:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o TEST.exe cnn.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog TEST.exe program.hex
	$(GCC_PREFIX)-objdump -S TEST.exe > TEST.dis

execute:
	whisper -x program.hex -s 0x80000000 --tohost 0xd0580000 -f log.txt --configfile whisper.json

parse: execute
	python3 parser.py

.PHONY: all clean compile execute parse
