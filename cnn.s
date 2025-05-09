.section .text              # Indicates this is executable code section
.globl _start             # Defines _start as a global symbol accessible outside this file

_start:                   # Main entry point of the program
    # Registers:
    # a0: input_matrix    # Will hold pointer to input image (likely 28x28)
    # a1: filter_kernel   # Will hold pointer to filter weights (5x5 kernels)
    # a2: output_filter   # Will hold pointer to where feature maps will be stored
    # a3: filter_bias     # Will hold pointer to filter bias values
    la   a0, input_matrix # Load address of input_matrix into a0
    la   a1, filter_kernel # Load address of filter kernels into a1
    la   a2, output_filter # Load address of output feature maps into a2
    la   a3, filter_bias   # Load address of filter biases into a3

    li   t0, 0             # Initialize filter index f = 0
conv_loop:                 # Start of convolution loop over 8 filters
    li   t1, 25            # Each filter has 25 elements (5x5 kernel)
    mul  t2, t0, t1        # t2 = f * 25 (elements)
    slli t2, t2, 2         # t2 = f * 25 * 4 (bytes) - shift left logical immediate by 2 (multiply by 4)
    add  a4, a1, t2        # a4 = address of current filter's kernel (filter_kernel + offset)

    li   t3, 576           # Each output feature map has 576 elements (24x24)
    mul  t4, t0, t3        # t4 = f * 576 (elements)
    slli t4, t4, 2         # t4 = f * 576 * 4 (bytes) - multiply by 4 for float size
    add  a5, a2, t4        # a5 = address where current filter's output will be stored

    slli t6, t0, 2         # t6 = f * 4 (bytes) - offset for current filter's bias
    add  a6, a3, t6        # a6 = address of current filter's bias value

    mv   a7, t0            # Pass filter index in a7 if needed by subroutines

    mv   a1, a4            # Move kernel address to a1 for conv2d function
    mv   a2, a5            # Move output address to a2 for conv2d function
    mv   a3, a6            # Move bias address to a3 for conv2d function
    call conv2d            # Call convolution function for this filter

    addi t0, t0, 1         # Increment filter index f++
    li   t5, 8             # We have 8 filters total
    blt  t0, t5, conv_loop # If f < 8, loop back to process next filter

    # Reuse a2 still pointing to output_filter
    call print             # Print the output of convolution operation

    # Maxpool each of the 8 output feature maps
    la   a0, output_filter # Load address of convolution output
    la   a1, output_pool   # Load address where pooled results will be stored

    li   t0, 0             # Initialize filter index f = 0
maxpool_loop:              # Start of maxpooling loop over 8 feature maps
    li   t1, 576           # Each feature map has 576 elements (24x24)
    mul  t2, t0, t1        # t2 = f * 576 (elements)
    slli t2, t2, 2         # t2 = f * 576 * 4 (bytes)
    add  a2, a0, t2        # a2 = address of current feature map

    li   t3, 144           # Each pooled output has 144 elements (12x12)
    mul  t4, t0, t3        # t4 = f * 144 (elements) 
    slli t4, t4, 2         # t4 = f * 144 * 4 (bytes)
    add  a3, a1, t4        # a3 = address where pooled output will be stored

    mv   a0, a2            # Move feature map address to a0 for maxpool function
    mv   a1, a3            # Move pooled output address to a1 for maxpool function
    call maxpool           # Call maxpooling function for this feature map

    addi t0, t0, 1         # Increment filter index f++
    li   t5, 8             # We have 8 feature maps to pool
    blt  t0, t5, maxpool_loop # If f < 8, loop back to process next feature map

    la   a0, output_pool   # Load address of pooled feature maps
    la   a1, weight_matrix # Load address of fully connected layer weights
    la   a2, bias_vector   # Load address of fully connected layer biases
    la   a3, final_output  # Load address where final output will be stored


    li   t0, 0             # Initialize classifier neuron index f = 0
    li   t1, 10            # We have 10 output neurons (likely digits 0-9)

    for_dense:             # Start of dense (fully connected) layer loop
    call denselayer        # Call dense layer computation for this neuron

    # increment weight matrix pointer by 1152 floats (4 * 1152 = 4608 bytes)
    li   t2, 4608          # Each output neuron has 1152 weights (12x12x8 inputs)
    add  a1, a1, t2        # Move to next neuron's weights

    # increment bias vector pointer by 1 float (4 bytes)
    addi a2, a2, 4         # Move to next neuron's bias

    # increment output pointer by 1 float (4 bytes)
    addi a3, a3, 4         # Move to next output position

    # increment loop counter
    addi t0, t0, 1         # Increment neuron index f++
    blt  t0, t1, for_dense # If f < 10, continue loop for next neuron

    
    
    la a0, final_output    # Load address of neural network raw outputs
    la a2, probability_matrix # Load address where softmax probabilities will be stored
  
    call softmax           # Apply softmax to convert raw outputs to probabilities

       


    j _finish              # Jump to finish routine

_finish:                   # Program termination routine
    li   t0, 0xd0500000    # Load special address for system control
    sb   zero, 0(t0)       # Store 0 to signal program termination
1:  j    1b                # Infinite loop as a fallback if termination fails

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

.globl conv2d              # Make conv2d function globally accessible
conv2d:                    # Start of convolution function
    li   t6, 5             # Kernel size is 5x5
    mv   s0, a0            # s0 = input matrix address
    mv   s2, a2            # s2 = output matrix address
    mv   s3, a1            # s3 = filter kernel address
    flw  f1, 0(a3)         # Load filter bias into f1

    li   t0, 0             # Initialize output row counter i = 0
.conv_i:                   # Start of loop over output rows
    li   t1, 0             # Initialize output column counter j = 0
    li   t6, 24            # Output size is 24x24
.conv_j:                   # Start of loop over output columns
    li   t4, 28            # Input width is 28
    mul  t5, t0, t4        # t5 = i * 28 (row offset in input)
    add  t5, t5, t1        # t5 = i * 28 + j (position in input)
    slli t5, t5, 2         # t5 = (i * 28 + j) * 4 (byte offset)
    add  t3, s0, t5        # t3 = address of input[i][j]

    mv   s4, s3            # s4 = filter kernel base address
    li   t2, 0             # Initialize filter row counter fi = 0
    li   t7, 5             # Filter size is 5x5
    vmv.v.i v4, 0          # Initialize accumulator vector to 0
.conv_fi:                  # Start of loop over filter rows
    vsetvli t4, t7, e32    # Set vector length for 32-bit elements
    vle32.v v1, (s4)       # Load filter row into v1 vector register
    vle32.v v0, (t3)       # Load input window row into v0 vector register
    vfmul.vv v2, v0, v1    # Multiply input and filter element-wise
    vfredosum.vs v4, v2, v4 # Accumulate sum of products into v4
    addi s4, s4, 20        # Move to next filter row (5 floats * 4 bytes)
    addi t3, t3, 112       # Move to next input row (28 floats * 4 bytes)
    addi t2, t2, 1         # Increment filter row counter
    blt  t2, t7, .conv_fi  # If not done with filter rows, continue

    vfmv.f.s f0, v4        # Move accumulated sum from vector to scalar
    fadd.s   f0, f0, f1    # Add bias to convolution result

    li   t6, 24            # Output width is 24
    mul  t5, t0, t6        # t5 = i * 24 (row offset in output)
    add  t5, t5, t1        # t5 = i * 24 + j (position in output)
    slli t5, t5, 2         # t5 = (i * 24 + j) * 4 (byte offset)
    add  t5, s2, t5        # t5 = address of output[i][j]
    fsw  f0, 0(t5)         # Store result in output matrix

    addi t1, t1, 1         # Increment column counter j++
    blt  t1, t6, .conv_j   # If j < 24, continue with next column

    addi t0, t0, 1         # Increment row counter i++
    blt  t0, t6, .conv_i   # If i < 24, continue with next row

    ret                    # Return from function

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

.globl maxpool             # Make maxpool function globally accessible
maxpool:                   # Start of maxpooling function
    li   t6, 12            # Output size is 12x12
    mv   s0, a0            # s0 = input feature map address
    mv   s1, a1            # s1 = output pooled map address
    li   t0, 0             # Initialize output row counter i = 0
.pool_i:                   # Start of loop over output rows
    li   t1, 0             # Initialize output column counter j = 0
.pool_j:                   # Start of loop over output columns
    slli t2, t0, 1         # t2 = i * 2 (input row)
    li   t3, 24            # Input width is 24
    mul  t2, t2, t3        # t2 = i * 2 * 24 (input row offset)
    slli t3, t1, 1         # t3 = j * 2 (input column)
    add  t2, t2, t3        # t2 = i * 2 * 24 + j * 2 (input position)
    slli t2, t2, 2         # t2 = (i * 2 * 24 + j * 2) * 4 (byte offset)
    add  t4, s0, t2        # t4 = address of input[i*2][j*2]
    addi t5, t4, 96        # t5 = address of input[i*2+1][j*2] (next row, 24*4=96)

    vsetvli t2, 2, e32     # Set vector length for 2 elements
    vle32.v v0, (t4)       # Load top 2x1 window into v0
    vle32.v v1, (t5)       # Load bottom 2x1 window into v1
    vmax.vv v2, v0, v1     # Find max between rows
    vmv.v.i v4, 0          # Initialize reduction vector
    vfredmax.vs v4, v2, v4 # Find maximum value in window
    vfmv.f.s f6, v4        # Move max value from vector to scalar

    mul  t3, t0, t6        # t3 = i * 12 (output row offset)
    add  t3, t3, t1        # t3 = i * 12 + j (output position)
    slli t3, t3, 2         # t3 = (i * 12 + j) * 4 (byte offset)
    add  t3, s1, t3        # t3 = address of output[i][j]
    fsw  f6, 0(t3)         # Store max value in output matrix

    addi t1, t1, 1         # Increment column counter j++
    blt  t1, t6, .pool_j   # If j < 12, continue with next column

    addi t0, t0, 1         # Increment row counter i++
    blt  t0, t6, .pool_i   # If i < 12, continue with next row

    ret                    # Return from function


   .globl denselayer       # Make denselayer function globally accessible
denselayer:                # Start of dense layer function
    mv s0, a0              # s0 = flattened input vector (1152 floats)
    mv s1, a1              # s1 = weights for this neuron (1152 floats)
    mv s2, a2              # s2 = bias for this neuron
    mv s3, a3              # s3 = output for this neuron

    li t0, 1152            # Total input elements to process (12x12x8)
    mv t3, s0              # t3 = input vector iterator
    mv t4, s1              # t4 = weight vector iterator

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

    ret                    # Return from function

# -----------------------------------------------
# Softmax: Applies softmax on 10-element vector
# Inputs:
#   a0 = input base ptr (final_output)
#   a1 = input ptr (looping through each element)
#   a2 = output ptr (probability_matrix)
# Assumes 10 elements


    .globl softmax         # Make softmax function globally accessible
softmax:                   # Start of softmax function
    # --- save arguments ---
    mv   s0, a0            # s0 = input array base address
    # mv   s1, a1          # s1 = number of elements (n) - commented out, not used
    mv   s1, a1            # Using s1 as working pointer (not following standard comment)
    mv   s2, a2            # s2 = output array base address

    # --- first pass: compute exp(x[i]) and accumulate sum in f1 ---
    li   t0, 0             # Initialize element counter i = 0
    fmv.s.x f1, zero       # f1 = 0.0 (sum accumulator for normalization)


    li t4, 10              # Size of output array (number of classes)
loop_exp:                  # Start of loop to calculate exponentials
    bge  t0, t4, do_norm   # If i ≥ n, jump to normalization
    slli t1, t0, 2         # t1 = i * 4 (byte offset)
    add  t2, s0, t1        # t2 = address of input[i]
    flw  fa0, 0(t2)        # fa0 = input[i] (raw score)
    call taylor_exp        # Call exponential function: fa0 = exp(input[i])
    # store exp(x[i]) to output[i]
    add  t3, s1, t1        # t3 = address of intermediate output[i]
    fsw  fa0, 0(t3)        # Store exponential at intermediate location
    # sum += exp(x[i])
    fadd.s f1, f1, fa0     # Add to sum for normalization

    addi t0, t0, 1         # Increment counter i++
    j    loop_exp          # Continue loop

    # --- second pass: normalize each exp(x[i]) by the sum ---
do_norm:                   # Start normalization phase
    li   t0, 0             # Reset counter i = 0

loop_norm:                 # Start of normalization loop
    bge  t0, t4, done      # If i ≥ n, we're done
    slli t1, t0, 2         # t1 = i * 4 (byte offset)
    add  t2, s1, t1        # t2 = address of intermediate exp(x[i])
    flw  fa0, 0(t2)        # fa0 = exp(x[i])
    fdiv.s fa0, fa0, f1    # fa0 = exp(x[i]) / sum (normalize)
    fsw  fa0, 0(t2)        # output[i] = softmax(x[i])

    addi t0, t0, 1         # Increment counter i++
    j    loop_norm         # Continue loop

done:                      # End of softmax function
    ret                    # Return from function

# -----------------------------------------------
# exp_approx: Approximate exp(f0) using a simple Taylor expansion
# Input:
#   fa0 = input float
# Output:
#   fa0 = exp(fa0)
# Approximation: 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 (Taylor series)
# -----------------------------------------------
.globl taylor_exp          # Make taylor_exp function globally accessible
taylor_exp:                # Start of exponential approximation function
    li t0, 1               # Load constant 1
    fcvt.s.w f0, t0        # Convert to float: f0 = 1.0 (accumulator)

    fmv.s f1, fa0          # f1 = x (input value)
    fmul.s f2, f1, f1      # f2 = x^2
    fmul.s f3, f2, f1      # f3 = x^3
    fmul.s f4, f3, f1      # f4 = x^4
    fmul.s f5, f4, f1      # f5 = x^5

    fadd.s f0, f0, f1      # f0 = 1 + x

    li   t0, 2             # Load constant 2
    fcvt.s.w f6, t0        # Convert to float: f6 = 2.0
    fdiv.s f2, f2, f6      # f2 = x^2 / 2

    fadd.s f0, f0, f2      # f0 = 1 + x + x^2/2

    li   t0, 6             # Load constant 6
    fcvt.s.w f7, t0        # Convert to float: f7 = 6.0
    fdiv.s f3, f3, f7      # f3 = x^3 / 6

    fadd.s f0, f0, f3      # f0 = 1 + x + x^2/2 + x^3/6

    li  t0, 24             # Load constant 24
    fcvt.s.w f8, t0        # Convert to float: f8 = 24.0
    fdiv.s f4, f4, f8      # f4 = x^4 / 24

    fadd.s f0, f0, f4      # f0 = 1 + x + x^2/2 + x^3/6 + x^4/24

    fmv.s fa0, f0          # Copy result to return register

    ret                    # Return from function

#----------------------------------------------------------------------
.globl print               # Make print function globally accessible
# Inputs:
#   a0 = vector pointer
#   a1 = vector size
print:                     # Start of print function (debug output)
    mv   t6, a1            # t6 = vector size
    mv   t3, a0            # t3 = vector address
    mv   t1, t6            # t1 = remaining elements
.print_loop:               # Start of print loop
    beq  t1, zero, .print_done # If no elements left, finish
    vsetvli t5, t1, e32    # Set vector length for remaining elements
    li t0, 1               # Debug instruction (no actual operation)
    li t0, 2               # Debug instruction (no actual operation)
    li t0, 3               # Debug instruction (no actual operation)

    vle32.v v0, (t3)       # Load vector chunk into v0 (not printed, just loaded)

    li t0, 1               # Debug instruction (no actual operation)
    li t0, 2               # Debug instruction (no actual operation)
    li t0, 3               # Debug instruction (no actual operation)

    slli t2, t5, 2         # t2 = processed elements * 4 (bytes)
    add  t3, t3, t2        # Advance vector pointer
    sub  t1, t1, t5        # Reduce remaining element count
    j    .print_loop       # Continue loop
.print_done:               # End of print function
    ret                    # Return from function