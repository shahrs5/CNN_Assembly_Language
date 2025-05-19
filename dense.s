.globl denselayer       # Make denselayer function globally accessible
denselayer:                # Start of dense layer function
    addi sp, sp, -32       # Allocate space on stack for t0–t6 (7 regs × 4 bytes)
    sw   t0, 0(sp)
    sw   t1, 4(sp)
    sw   t2, 8(sp)
    sw   t3, 12(sp)
    sw   t4, 16(sp)
    sw   t5, 20(sp)
    sw   t6, 24(sp)
    sw   ra, 28(sp)
    mv s0, a0              # s0 = flattened input vector (1152 floats)
    mv s1, a1              # s1 = weights for this neuron (1152 floats)
    mv s2, a2              # s2 = bias for this neuron
    mv s3, a3              # s3 = output for this neuron

    li t0, 1152            # Total input elements to process (12x12x8)
    mv t3, s0              # t3 = input vector iterator
    mv t4, s1              # t4 = weight vector iterator

    li t5 8
    vsetvli zero, t5, e32
    vmv.v.i v4, 0          # Initialize accumulator vector to 0

        .dense_loop:               # Start of loop processing input chunks
            vsetvli t1, t0, e32    # Set vector length based on remaining elements
            vle32.v v0, (t3)       # Load input vector chunk
            vle32.v v1, (t4)       # Load weight vector chunk
            vfmul.vv v2, v0, v1    # Element-wise multiply inputs and weights
            vfredosum.vs v4, v2, v4 # Accumulate dot product into v4

            slli t2, t1, 2         # t2 = processed elements * 4 (bytes)
            add t3, t3, t2         # Advance input pointer
            add t4, t4, t2         # Advance weight pointer
            sub t0, t0, t1         # Reduce remaining element count
        bnez t0, .dense_loop   # Loop if elements remain

    vfmv.f.s f0, v4        # Move accumulated sum from vector to scalar
    flw f1, 0(s2)          # Load bias value
    fadd.s f2, f0, f1      # Add bias to weighted sum
    fsw f2, 0(s3)          # Store result in output neuron