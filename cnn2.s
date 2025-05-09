.section .data
.align 4

input_matrix:
    .rept 784             # 28×28 floats
      .float 0.0
    .endr

filter_kernel:
    .rept 200             # 5×5×8 = 200 floats
      .float 0.0
    .endr

filter_bias:
    .rept 8               # 8 floats
        .float 0.0
    .endr

output_matrix:
    .rept 4608            # 24×24×8
      .float 0.0
    .endr

output_matrix2:
    .rept 1152            # 12×12×8
      .float 0.0
    .endr

weight_matrix:
    .rept 11520           # 10×1152
        .float 0.0
    .endr

bias_vector:
     .rept 10              # intermediate output vector
        .float 0.0
    .endr

output_matrix3:
    .rept 10              # final output vector
        .float 0.0
    .endr

probability_matrix:
    .rept 10
        .float 0.0
    .endr









output_matrix3:
    .rept 10              # intermediate output vector
        .float 0.0
    .endr

.section .text
.globl _start

_start:
    # Registers:
    # a0: input_matrix
    # a1: filter_kernel (offset per filter)
    # a2: output_matrix (offset per feature map)
    # a3: filter_bias
    la   a0, input_matrix
    la   a1, filter_kernel
    la   a2, output_matrix
    la   a3, filter_bias

    li   t0, 0             # filter index f = 0
conv_loop:
    li   t1, 25
    mul  t2, t0, t1
    slli t2, t2, 2         # offset = f * 25 * 4
    add  a4, a1, t2        # a4 = kernel for filter f

    li   t3, 576
    mul  t4, t0, t3
    slli t4, t4, 2
    add  a5, a2, t4        # a5 = output base for filter f

    slli t6, t0, 2
    add  a6, a3, t6        # a6 = bias[f]

    mv   a7, t0            # pass filter index if needed

    mv   a1, a4
    mv   a2, a5
    mv   a3, a6
    call conv2d

    addi t0, t0, 1
    li   t5, 8
    blt  t0, t5, conv_loop

    # Reuse a2 still pointing to output_matrix
    call print

    # Maxpool each of the 8 output feature maps
    la   a0, output_matrix
    la   a1, output_matrix2

    li   t0, 0
maxpool_loop:
    li   t1, 576
    mul  t2, t0, t1
    slli t2, t2, 2
    add  a2, a0, t2        # input pointer

    li   t3, 144
    mul  t4, t0, t3
    slli t4, t4, 2
    add  a3, a1, t4        # output pointer

    mv   a0, a2
    mv   a1, a3
    call maxpool

    addi t0, t0, 1
    li   t5, 8
    blt  t0, t5, maxpool_loop

    la   a0, output_matrix2
    la   a1, weight_matrix
    la   a2 , bias_vector
    la   a3,output_matrix3


    li   t0, 0           # f = 0
    li   t1, 10          # loop upper bound

    for_dense:
    call denselayer

    # increment weight matrix pointer by 1152 floats (4 * 1152 = 4608 bytes)
    li   t2, 4608
    add  a1, a1, t2

    # increment bias vector pointer by 1 float (4 bytes)
    addi a2, a2, 4

    # increment output pointer by 1 float (4 bytes)
    addi a3, a3, 4

    # increment loop counter
    addi t0, t0, 1
    blt  t0, t1, for_dense

    
    
    la a0 ,output_matrix3
    la a2 ,probability_matrix
  
    call softmax

       


    j _finish

_finish:
    li   t0, 0xd0500000
    sb   zero, 0(t0)
1:  j    1b

#----------------------------------------------------------------------
# conv2d: Convolution 5x5 on 28x28 → 24x24 output
# Inputs:
#   a0 = input base ptr
#   a1 = filter base ptr (25 floats)
#   a2 = output base ptr
#   a3 = bias ptr (float *)
# Registers used:
#   s0 = input ptr
#   s1 = not used
#   s2 = output ptr
#   s3 = filter ptr
#   s4 = temp filter row ptr
#   t0-t6 = loop counters, offsets
#   v0-v4 = vector registers for computation

.globl conv2d
conv2d:
    li   t6, 5
    mv   s0, a0
    mv   s2, a2
    mv   s3, a1
    flw  f1, 0(a3)     # load filter bias

    li   t0, 0
.conv_i:
    li   t1, 0
    li   t6, 24
.conv_j:
    li   t4, 28
    mul  t5, t0, t4
    add  t5, t5, t1
    slli t5, t5, 2
    add  t3, s0, t5    # t3 = &input[i][j]

    mv   s4, s3        # s4 = filter base
    li   t2, 0
    li   t7, 5
    vmv.v.i v4, 0
.conv_fi:
    vsetvli t4, t7, e32
    vle32.v v1, (s4)
    vle32.v v0, (t3)
    vfmul.vv v2, v0, v1
    vfredosum.vs v4, v2, v4
    addi s4, s4, 20
    addi t3, t3, 112
    addi t2, t2, 1
    blt  t2, t7, .conv_fi

    vfmv.f.s f0, v4
    fadd.s   f0, f0, f1  # add bias

    li   t6, 24
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

#----------------------------------------------------------------------
# maxpool: 2x2 max pool on 24x24 input to 12x12 output
# Inputs:
#   a0 = input base ptr (24x24)
#   a1 = output base ptr (12x12)
# Registers:
#   s0 = input ptr
#   s1 = output ptr
#   t0 = row i
#   t1 = col j
#   v0, v1 = rows
#   v2 = max
#   v4 = reduced max

.globl maxpool
maxpool:
    li   t6, 12
    mv   s0, a0
    mv   s1, a1
    li   t0, 0
.pool_i:
    li   t1, 0
.pool_j:
    slli t2, t0, 1
    li   t3, 24
    mul  t2, t2, t3
    slli t3, t1, 1
    add  t2, t2, t3
    slli t2, t2, 2
    add  t4, s0, t2
    addi t5, t4, 96

    vsetvli t2, 2, e32
    vle32.v v0, (t4)
    vle32.v v1, (t5)
    vmax.vv v2, v0, v1
    vmv.v.i v4, 0
    vfredmax.vs v4, v2, v4
    vfmv.f.s f6, v4

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


   .globl denselayer
denselayer:
    mv s0, a0        # input vector ptr (1152 floats)
    mv s1, a1        # weight vector ptr (1152 floats)
    mv s2, a2        # bias[i] ptr
    mv s3, a3        # output[i] ptr

    li t0, 1152      # total elements to process
    mv t3, s0        # input vector iterator
    mv t4, s1        # weight vector iterator

    vmv.v.i v4, 0    # accumulator initialized to 0

.dense_loop:
    vsetvli t1, t0, e32      # set vector length based on remaining elements
    vle32.v v0, (t3)         # load input vector chunk
    vle32.v v1, (t4)         # load weight vector chunk
    vfmul.vv v2, v0, v1      # element-wise multiply
    vfredosum.vs v4, v2, v4  # accumulate partial sum into v4

    slli t2, t1, 2           # bytes = elements * 4
    add t3, t3, t2           # advance input pointer
    add t4, t4, t2           # advance weight pointer
    sub t0, t0, t1           # reduce remaining count
    bnez t0, .dense_loop     # loop if not done

    vfmv.f.s f0, v4          # move sum from v4 to scalar f0
    flw f1, 0(s2)            # load bias[i]
    fadd.s f2, f0, f1        # add bias
    fsw f2, 0(s3)            # store result in output[i]

    ret

# -----------------------------------------------
# Softmax: Applies softmax on 10-element vector
# Inputs:
#   a0 = input base ptr (output_matrix3)
#   a1 = input ptr (looping through each element)
#   a2 = output ptr (probability_matrix)
# Assumes 10 elements




# .globl softmax
# softmax:
#     mv s0 ,a0 #address of input
#     mv s1 ,a1 # address of output



#     # make a for loop untill vector is exhausted
#     li t1 ,10 # input vector size
#     vsetvli t1, t0, e32
#     vle32.v v1, (s0)
#     vfredmul v2,v1 #reduce by multiplication
#     #move v2 to f0
#     # end loop



#     mv fa0 ,s1 #move the data at address s1 to fa0 theb call taylor_exp(fa0)

#     call taylor_exp
#     #taylor-exp return value in fa0

#     # divide f0 by fa0 and store in f0
#     #store f0 at address s2





    
  
#     ret



    .globl softmax
softmax:
    # --- save arguments ---
    mv   s0, a0        # s0 ← input array base address
    # mv   s1, a1        # s1 ← number of elements (n)
    mv   s1, a1        # s2 ← output array base address

    # --- first pass: compute exp(x[i]) and accumulate sum in f1 ---
    li   t0, 0         # t0 = i = 0
    fmv.s.x f1, zero   # f1 = 0.0 (sum accumulator)


    li t4 ,10 #size of array
loop_exp:
    bge  t0, t4, do_norm   # if i ≥ n, jump to normalization
    slli t1, t0, 2         # t1 = byte offset = i * 4
    add  t2, s0, t1        # t2 = &input[i]
    flw  fa0, 0(t2)        # fa0 = input[i]
    call taylor_exp        # → fa0 = exp(input[i])
    # store exp(x[i]) to output[i]
    add  t3, s1, t1        # t3 = &output[i]
    fsw  fa0, 0(t3)
    # sum += exp(x[i])
    fadd.s f1, f1, fa0

    addi t0, t0, 1
    j    loop_exp

    # --- second pass: normalize each exp(x[i]) by the sum ---
do_norm:
    li   t0, 0             # reset i = 0

loop_norm:
    bge  t0, t4, done      # if i ≥ n, we’re done
    slli t1, t0, 2         # byte offset = i * 4
    add  t2, s1, t1        # t2 = &output[i]
    flw  fa0, 0(t2)        # fa0 = exp(x[i])
    fdiv.s fa0, fa0, f1    # fa0 = exp(x[i]) / sum
    fsw  fa0, 0(t2)        # output[i] = softmax(x[i])

    addi t0, t0, 1
    j    loop_norm

done:
    ret







# -----------------------------------------------
# exp_approx: Approximate exp(f0) using a simple Taylor expansion
# Input:
#   f0 = input float
# Output:
#   f0 = exp(f0)
# Approximation: 1 + x + x^2/2 + x^3/6 (good for small x)
# -----------------------------------------------
.globl taylor_exp
.taylor_exp:
    li t0 ,1
    fcvt.s.w f0,t0 #accumulator

    fmv.s f1, fa0        # f1 = x
    fmul.s f2, f1, f1   # f2 = x^2
    fmul.s f3, f2, f1   # f3 = x^3
    fmul.s f4, f3, f1   # f3 = x^4
    fmul.s f5, f4, f1   # f3 = x^5

    fadd.s f0, f0, f1

    li   t0, 2
    fcvt.s.w f6, t0
    fdiv.s f2, f2, f6   # x^2 / 2

    fadd.s f0,f0,f2



    li   t0, 6
    fcvt.s.w f7, t0
    fdiv.s f3, f3, f7   # x^3 / 6

    fadd.s f0,f0, f3


    li  t0, 24
    fcvt.s.w f8, t0
    fdiv.s f4, f4, f8   # x^4 / 24

    fadd.s f0,f0,f4

    fmv.s fa0,f0

    # li   t0, 1
    # fcvt.s.w f6, t0     # 1.0

    # fadd.s f0, f6, f1   # f0 = 1 + x
    # fadd.s f0, f0, f2   # f0 += x^2/2
    # fadd.s f0, f0, f3   # f0 += x^3/6

    ret


#----------------------------------------------------------------------
.globl print
# Inputs:
#   a0 = vector pointer
#   a1 = vector size
print:
    mv   t6, a1
    mv   t3, a0
    mv   t1, t6
.print_loop:
    beq  t1, zero, .print_done
    vsetvli t5, t1, e32
    li t0 ,1
    li t0 ,2
    li t0, 3

    vle32.v v0, (t3)


    li t0 ,1
    li t0 ,2
    li t0, 3

    slli t2, t5, 2
    add  t3, t3, t2
    sub  t1, t1, t5
    j    .print_loop
.print_done:
    ret
