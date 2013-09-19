package org.apfloat.internal;

import java.math.BigInteger;
import java.util.RandomAccess;

import org.apfloat.ApfloatContext;
import org.apfloat.ApfloatRuntimeException;
import org.apfloat.spi.DataStorageBuilder;
import org.apfloat.spi.DataStorage;
import static org.apfloat.internal.DoubleModConstants.*;

/**
 * Class for performing the final step of a three-modulus
 * Number Theoretic Transform based convolution. Works for the
 * <code>double</code> type.<p>
 *
 * The algorithm is parallelized for multiprocessor computers.
 * The parallelization works so that the carry-CRT is done in
 * blocks in parallel. As a final step, a second pass is done
 * through the data set to propagate the carries from one block
 * to the next.
 *
 * @version 1.6.1
 * @author Mikko Tommila
 */

public class DoubleCarryCRT
    extends DoubleCRTMath
{
    // Runnable for calculating the carry-CRT in blocks, then get the carry of the previous block, add it, and provide carry to the next block
    private class CarryCRTRunnable
        implements Runnable
    {
        public CarryCRTRunnable(DataStorage resultMod0, DataStorage resultMod1, DataStorage resultMod2, DataStorage dataStorage, long size, long resultSize, long offset, long length, MessagePasser<Long, double[]> messagePasser)
        {
            this.resultMod0 = resultMod0;
            this.resultMod1 = resultMod1;
            this.resultMod2 = resultMod2;
            this.dataStorage = dataStorage;
            this.size = size;
            this.resultSize = resultSize;
            this.offset = offset;
            this.length = length;
            this.messagePasser = messagePasser;
        }

        public void run()
        {
            long skipSize = (this.offset == 0 ? this.size - this.resultSize + 1: 0);    // For the first block, ignore the first 1-3 elements
            long lastSize = (this.offset + this.length == this.size ? 1: 0);            // For the last block, add 1 element
            long nonLastSize = 1 - lastSize;                                            // For the other than last blocks, move 1 element
            long subResultSize = this.length - skipSize + lastSize;

            long subStart = this.size - this.offset,
                 subEnd = subStart - this.length,
                 subResultStart = this.size - this.offset - this.length + nonLastSize + subResultSize,
                 subResultEnd = subResultStart - subResultSize;

            DataStorage.Iterator src0 = this.resultMod0.iterator(DataStorage.READ, subStart, subEnd),
                                 src1 = this.resultMod1.iterator(DataStorage.READ, subStart, subEnd),
                                 src2 = this.resultMod2.iterator(DataStorage.READ, subStart, subEnd),
                                 dst = this.dataStorage.iterator(DataStorage.WRITE, subResultStart, subResultEnd);

            double[] carryResult = new double[3],
                      sum = new double[3],
                      tmp = new double[3];

            // Preliminary carry-CRT calculation (happens in parallel in multiple blocks)
            for (long i = 0; i < this.length; i++)
            {
                double y0 = MATH_MOD_0.modMultiply(T0, src0.getDouble()),
                        y1 = MATH_MOD_1.modMultiply(T1, src1.getDouble()),
                        y2 = MATH_MOD_2.modMultiply(T2, src2.getDouble());

                multiply(M12, y0, sum);
                multiply(M02, y1, tmp);

                if (add(tmp, sum) != 0 ||
                    compare(sum, M012) >= 0)
                {
                    subtract(M012, sum);
                }

                multiply(M01, y2, tmp);

                if (add(tmp, sum) != 0 ||
                    compare(sum, M012) >= 0)
                {
                    subtract(M012, sum);
                }

                add(sum, carryResult);

                double result = divide(carryResult);

                // In the first block, ignore the first element (it's zero in full precision calculations)
                // and possibly one or two more in limited precision calculations
                if (i >= skipSize)
                {
                    dst.setDouble(result);
                    dst.next();
                }

                src0.next();
                src1.next();
                src2.next();
            }

            // Calculate the last words (in base math)
            double result0 = divide(carryResult);
            double result1 = carryResult[2];

            assert (carryResult[0] == 0);
            assert (carryResult[1] == 0);

            // Last block has one extra element (corresponding to the one skipped in the first block)
            if (subResultSize == this.length - skipSize + 1)
            {
                dst.setDouble(result0);
                dst.close();

                result0 = result1;
                assert (result1 == 0);
            }

            double[] results = { result1, result0 };

            // Finishing step - get the carry from the previous block and propagate it through the data
            if (this.offset > 0)
            {
                double[] previousResults = this.messagePasser.receiveMessage(this.offset);

                // Get iterators for the previous block carries, and dst, padded with this block's carries
                // Note that size could be 1 but carries size is 2
                DataStorage.Iterator src = arrayIterator(previousResults);
                dst = compositeIterator(this.dataStorage.iterator(DataStorage.READ_WRITE, subResultStart, subResultEnd), subResultSize, arrayIterator(results));

                // Propagate base addition through dst, and this block's carries
                double carry = baseAdd(dst, src, 0, dst, previousResults.length);
                carry = baseCarry(dst, carry, subResultSize);
                dst.close();                                                    // Iterator likely was not iterated to end

                assert (carry == 0);
            }

            // Finally, send the carry to the next block
            this.messagePasser.sendMessage(this.offset + this.length, results);
        }

        private double baseCarry(DataStorage.Iterator srcDst, double carry, long size)
        {
            for (long i = 0; i < size && carry > 0; i++)
            {
                carry = baseAdd(srcDst, null, carry, srcDst, 1);
            }

            return carry;
        }

        private DataStorage resultMod0,
                            resultMod1,
                            resultMod2,
                            dataStorage;
        private long size,
                     resultSize,
                     offset,
                     length;
        private MessagePasser<Long, double[]> messagePasser;
    }

    /**
     * Creates a carry-CRT object using the specified radix.
     *
     * @param radix The radix that will be used.
     */

    public DoubleCarryCRT(int radix)
    {
        super(radix);
    }

    /**
     * Calculate the final result of a three-NTT convolution.<p>
     *
     * Performs a Chinese Remainder Theorem (CRT) on each element
     * of the three result data sets to get the result of each element
     * modulo the product of the three moduli. Then it calculates the carries
     * to get the final result.<p>
     *
     * Note that the return value's initial word may be zero or non-zero,
     * depending on how large the result is.<p>
     *
     * Assumes that <code>MODULUS[0] > MODULUS[1] > MODULUS[2]</code>.
     *
     * @param resultMod0 The result modulo <code>MODULUS[0]</code>.
     * @param resultMod1 The result modulo <code>MODULUS[1]</code>.
     * @param resultMod2 The result modulo <code>MODULUS[2]</code>.
     * @param resultSize The number of elements needed in the final result.
     *
     * @return The final result with the CRT performed and the carries calculated.
     */

    public DataStorage carryCRT(final DataStorage resultMod0, final DataStorage resultMod1, final DataStorage resultMod2, final long resultSize)
        throws ApfloatRuntimeException
    {
        final long size = Math.min(resultSize + 2, resultMod0.getSize());   // Some extra precision if not full result is required

        ApfloatContext ctx = ApfloatContext.getContext();
        DataStorageBuilder dataStorageBuilder = ctx.getBuilderFactory().getDataStorageBuilder();
        final DataStorage dataStorage = dataStorageBuilder.createDataStorage(resultSize * 8);
        dataStorage.setSize(resultSize);

        final MessagePasser<Long, double[]> messagePasser = new MessagePasser<Long, double[]>();

        if (size <= Integer.MAX_VALUE && this.parallelRunner != null &&     // Only if the size fits in an integer, but with memory arrays it should
            resultMod0 instanceof RandomAccess &&                           // Only if the data storage supports efficient parallel random access
            resultMod1 instanceof RandomAccess &&
            resultMod2 instanceof RandomAccess &&
            dataStorage instanceof RandomAccess)
        {
            ParallelRunnable parallelRunnable = new ParallelRunnable()
            {
                public int getLength()
                {
                    return (int) size;
                }

                public Runnable getRunnable(int offset, int length)
                {
                    return new CarryCRTRunnable(resultMod0, resultMod1, resultMod2, dataStorage, size, resultSize, offset, length, messagePasser);
                }
            };

            this.parallelRunner.runParallel(parallelRunnable);
        }
        else
        {
            // Just run in current thread without parallelization
            new CarryCRTRunnable(resultMod0, resultMod1, resultMod2, dataStorage, size, resultSize, 0, size, messagePasser).run();
        }

        // Sanity check
        double[] carries = null;
        assert ((carries = messagePasser.getMessage(size)) != null);
        assert (carries.length == 2);
        assert (carries[0] == 0);
        assert (carries[1] == 0);

        return dataStorage;
    }

    /**
     * Set the parallel runner to be used when executing the CRT.
     *
     * @param parallelRunner The parallel runner.
     */

    public void setParallelRunner(ParallelRunner parallelRunner)
    {
        this.parallelRunner = parallelRunner;
    }

    // Wrap an array in a simple reverse-order iterator, padded with zeros
    private static DataStorage.Iterator arrayIterator(final double[] data)
    {
        return new DataStorage.Iterator()
        {
            public boolean hasNext()
            {
                return true;
            }

            public void next()
            {
                this.position--;
            }

            public double getDouble()
            {
                assert (this.position >= 0);
                return data[this.position];
            }

            public void setDouble(double value)
            {
                assert (this.position >= 0);
                data[this.position] = value;
            }

            private int position = data.length - 1;
        };
    }

    // Composite iterator, made by concatenating two iterators
    private static DataStorage.Iterator compositeIterator(final DataStorage.Iterator iterator1, final long size, final DataStorage.Iterator iterator2)
    {
        return new DataStorage.Iterator()
        {
            public boolean hasNext()
            {
                return (this.position < size ? iterator1.hasNext() : iterator2.hasNext());
            }

            public void next()
            {
                (this.position < size ? iterator1 : iterator2).next();
                this.position++;
            }

            public double getDouble()
            {
                return (this.position < size ? iterator1 : iterator2).getDouble();
            }

            public void setDouble(double value)
            {
                (this.position < size ? iterator1 : iterator2).setDouble(value);
            }

            public void close()
                throws ApfloatRuntimeException
            {
                (this.position < size ? iterator1 : iterator2).close();
            }

            private long position;
        };
    }

    private static final long serialVersionUID = -3954870352092656433L;

    private static final DoubleModMath MATH_MOD_0,
                                        MATH_MOD_1,
                                        MATH_MOD_2;
    private static final double T0,
                                 T1,
                                 T2;
    private static final double[] M01,
                                   M02,
                                   M12,
                                   M012;

    private ParallelRunner parallelRunner;

    static
    {
        MATH_MOD_0 = new DoubleModMath();
        MATH_MOD_1 = new DoubleModMath();
        MATH_MOD_2 = new DoubleModMath();

        MATH_MOD_0.setModulus(MODULUS[0]);
        MATH_MOD_1.setModulus(MODULUS[1]);
        MATH_MOD_2.setModulus(MODULUS[2]);

        // Probably sub-optimal, but it's a one-time operation

        BigInteger base = BigInteger.valueOf(Math.abs((long) MAX_POWER_OF_TWO_BASE)),   // In int case the base is 0x80000000
                   m0 = BigInteger.valueOf((long) MODULUS[0]),
                   m1 = BigInteger.valueOf((long) MODULUS[1]),
                   m2 = BigInteger.valueOf((long) MODULUS[2]),
                   m01 = m0.multiply(m1),
                   m02 = m0.multiply(m2),
                   m12 = m1.multiply(m2);

        T0 = m12.modInverse(m0).doubleValue();
        T1 = m02.modInverse(m1).doubleValue();
        T2 = m01.modInverse(m2).doubleValue();

        M01 = new double[2];
        M02 = new double[2];
        M12 = new double[2];
        M012 = new double[3];

        BigInteger[] qr = m01.divideAndRemainder(base);
        M01[0] = qr[0].doubleValue();
        M01[1] = qr[1].doubleValue();

        qr = m02.divideAndRemainder(base);
        M02[0] = qr[0].doubleValue();
        M02[1] = qr[1].doubleValue();

        qr = m12.divideAndRemainder(base);
        M12[0] = qr[0].doubleValue();
        M12[1] = qr[1].doubleValue();

        qr = m0.multiply(m12).divideAndRemainder(base);
        M012[2] = qr[1].doubleValue();
        qr = qr[0].divideAndRemainder(base);
        M012[0] = qr[0].doubleValue();
        M012[1] = qr[1].doubleValue();
    }
}
