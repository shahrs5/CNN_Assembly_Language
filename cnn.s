    .section .data
    .align 4

input_matrix:
    .rept 784             # 28×28 floats
      .float 0.0
    .endr

filter_kernel:
    .rept 25              # 5×5 floats
      .float 0.0
    .endr

output_matrix:
    .rept 576             # 24×24 floats
      .float 0.0
    .endr

output_matrix2:
    .rept 144             # 12×12 floats
      .float 0.0
    .endr


    .section .text
    .globl _start

_start:
    #── conv2d ─────────────────────────────────────────────────────
    la   a0, input_matrix    # arg0
    la   a1, filter_kernel   # arg1
    la   a2, output_matrix   # arg2
    call conv2d

    #── print ───────────────────────────────────────────────────────
    # reuse a2 → still points at output_matrix
    call print

    #── maxpool ────────────────────────────────────────────────────
    la   a0, output_matrix   # arg0
    la   a1, output_matrix2  # arg1
    call maxpool

    #── finish / hang ─────────────────────────────────────────────
_finish:
    li   t0, 0xd0500000
    sb   zero, 0(t0)
1:  j    1b


    #-------------------------------------------
    .globl conv2d
# void conv2d(float *in, float *flt, float *out)
conv2d:
    # t6 = filter width = 5
    li   t6, 5
    mv   s0, a0        # s0 = input base
    mv   s3, a1        # s3 = filter base
    mv   s2, a2        # s2 = output base

    li   t0, 0         # i = 0
.conv_i:
    li   t1, 0         # j = 0
    # reload output width into t6
    li   t6, 24        # output width

.conv_j:
    # compute &in[i][j] → t3
    li   t4, 28
    mul  t5, t0, t4
    add  t5, t5, t1
    slli t5, t5, 2
    add  t3, s0, t5

    mv   s4, s3       # s4 = filter row ptr
    li   t2, 0        # fi = 0
    # set t6 back to filter width for inner loop
    li   t6, 5

.conv_fi:
    # v1 ← filter[fi][0..4]
    vsetvli t4, t6, e32
    vle32.v  v1, (s4)

    # v0 ← in[i+fi][j..j+4]
    vle32.v  v0, (t3)

    # v2 = v0 * v1; sum into v4
    vfmul.vv    v2, v0, v1
    vmv.v.i     v4, 0
    vfredosum.vs v4, v2, v4
    vfmv.f.s    f0, v4

    # advance filter ptr by 5 floats (20 bytes)
    addi s4, s4, 20
    # advance input ptr by one row (28 floats = 112 bytes)
    addi t3, t3, 112

    addi t2, t2, 1
    blt  t2, t6, .conv_fi

    # now t6 still = 5; reload output width into t6
    li   t6, 24

    # store f0 → output[i][j]
    mul  t5, t0, t6
    add  t5, t5, t1
    slli t5, t5, 2
    add  t5, s2, t5
    fsw  f0, 0(t5)

    addi t1, t1, 1
    blt  t1, t6, .conv_j

    addi t0, t0, 1
    blt  t0, t6, .conv_i

    ret


    #-------------------------------------------
    .globl maxpool
# void maxpool(float *in24x24, float *out12x12)
maxpool:
    # t6 = output dim = 12
    li   t6, 12
    mv   s0, a0        # s0 = in base
    mv   s1, a1        # s1 = out base

    li   t0, 0         # i = 0
.pool_i:
    li   t1, 0         # j = 0

.pool_j:
    # compute &in[2*i][2*j] → t4
    slli t2, t0, 1     # 2*i
    li   t3, 24
    mul  t2, t2, t3
    slli t3, t1, 1     # 2*j
    add  t2, t2, t3
    slli t2, t2, 2
    add  t4, s0, t2    # row0 ptr

    addi t5, t4, 96    # row1 ptr

    # load 2 lanes from each row
    vsetvli t2, t6, e32   # t6=12 → VL=2 actually since only 2 elements valid
    vle32.v  v0, (t4)
    vle32.v  v1, (t5)

    # vmax + reduce-max → f6
    vmax.vv      v2, v0, v1
    vmv.v.i      v4, 0
    vfredmax.vs  v4, v2, v4
    vfmv.f.s     f6, v4

    # store to out[i][j]
    mul  t3, t0, t6
    add  t3, t3, t1
    slli t3, t3, 2
    add  t3, s1, t3
    fsw  f6, 0(t3)

    addi t1, t1, 1
    blt  t1, t6, .pool_j

    addi t0, t0, 1
    blt  t0, t6, .pool_i

    ret


    #-------------------------------------------
    .globl print
# void print(float *out576)
print:
    # t6 = total elements = 576
    li   t6, 576
    mv   t3, a2        # t3 = ptr
    mv   t1, t6        # t1 = count

.print_loop:
    beq  t1, zero, .print_done

    vsetvli t5, t1, e32
    vle32.v  v0, (t3)

    # advance pointer by vl*4
    slli t2, t5, 2
    add  t3, t3, t2

    sub  t1, t1, t5
    j    .print_loop

.print_done:
    ret
