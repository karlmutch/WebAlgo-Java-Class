package org.apfloat.internal;

import org.apfloat.ApfloatRuntimeException;
import org.apfloat.spi.NTTStrategy;
import org.apfloat.spi.DataStorage;
import org.apfloat.spi.Util;
import static org.apfloat.internal.IntModConstants.*;

/**
 * A transform that implements a 3-point transform on
 * top of another Number Theoretic Transform that does
 * transforms of length 2<sup>n</sup>.
 *
 * @version 1.5.1
 * @author Mikko Tommila
 */

public class IntFactor3NTTStrategy
    extends IntModMath
    implements ParallelNTTStrategy
{
    /**
     * Creates a new factor-3 transform strategy on top of an existing transform.
     * The underlying transform needs to be capable of only doing transforms of
     * length 2<sup>n</sup>.
     *
     * @param factor2Strategy The underlying transformation strategy, that can be capable of only doing radix-2 transforms.
     */

    public IntFactor3NTTStrategy(NTTStrategy factor2Strategy)
    {
        this.factor2Strategy = factor2Strategy;
    }

    public void setParallelRunner(ParallelRunner parallelRunner)
    {
        // This may be sub-optimal; ideally this class should only implement ParallelNTTStrategy if the underlying transform does
        if (this.factor2Strategy instanceof ParallelNTTStrategy)
        {
            ((ParallelNTTStrategy) this.factor2Strategy).setParallelRunner(parallelRunner);
        }
    }

    public void transform(DataStorage dataStorage, int modulus)
        throws ApfloatRuntimeException
    {
        long length = dataStorage.getSize(),
             power2length = (length & -length);

        if (length > MAX_TRANSFORM_LENGTH)
        {
            throw new TransformLengthExceededException("Maximum transform length exceeded: " + length + " > " + MAX_TRANSFORM_LENGTH);
        }

        if (length == power2length)
        {
            // Transform length is a power of two
            this.factor2Strategy.transform(dataStorage, modulus);
        }
        else
        {
            // Transform length is three times a power of two
            assert (length == 3 * power2length);

            setModulus(MODULUS[modulus]);                                       // Modulus
            int w = getForwardNthRoot(PRIMITIVE_ROOT[modulus], length),     // Forward n:th root
                    w3 = modPow(w, (int) power2length);                     // Forward 3rd root

            DataStorage dataStorage0 = dataStorage.subsequence(0, power2length),
                        dataStorage1 = dataStorage.subsequence(power2length, power2length),
                        dataStorage2 = dataStorage.subsequence(2 * power2length, power2length);

            // Transform the columns
            transformColumns(false, dataStorage0, dataStorage1, dataStorage2, power2length, w, w3);

            // Transform the rows
            this.factor2Strategy.transform(dataStorage0, modulus);
            this.factor2Strategy.transform(dataStorage1, modulus);
            this.factor2Strategy.transform(dataStorage2, modulus);
        }
    }

    public void inverseTransform(DataStorage dataStorage, int modulus, long totalTransformLength)
        throws ApfloatRuntimeException
    {
        long length = dataStorage.getSize(),
             power2length = (length & -length);

        if (Math.max(length, totalTransformLength) > MAX_TRANSFORM_LENGTH)
        {
            throw new TransformLengthExceededException("Maximum transform length exceeded: " + Math.max(length, totalTransformLength) + " > " + MAX_TRANSFORM_LENGTH);
        }

        if (length == power2length)
        {
            // Transform length is a power of two
            this.factor2Strategy.inverseTransform(dataStorage, modulus, totalTransformLength);
        }
        else
        {
            // Transform length is three times a power of two
            assert (length == 3 * power2length);

            setModulus(MODULUS[modulus]);                                       // Modulus
            int w = getInverseNthRoot(PRIMITIVE_ROOT[modulus], length),     // Inverse n:th root
                    w3 = modPow(w, (int) power2length);                     // Inverse 3rd root

            DataStorage dataStorage0 = dataStorage.subsequence(0, power2length),
                        dataStorage1 = dataStorage.subsequence(power2length, power2length),
                        dataStorage2 = dataStorage.subsequence(2 * power2length, power2length);

            // Transform the rows
            this.factor2Strategy.inverseTransform(dataStorage0, modulus, totalTransformLength);
            this.factor2Strategy.inverseTransform(dataStorage1, modulus, totalTransformLength);
            this.factor2Strategy.inverseTransform(dataStorage2, modulus, totalTransformLength);

            // Transform the columns
            transformColumns(true, dataStorage0, dataStorage1, dataStorage2, power2length, w, w3);
        }
    }

    public long getTransformLength(long size)
    {
        // Calculates the needed transform length, that is
        // a power of two, or three times a power of two
        return Util.round23up(size);
    }

    // Transform the columns using a 3-point transform
    private void transformColumns(boolean isInverse, DataStorage dataStorage0, DataStorage dataStorage1, DataStorage dataStorage2, long size, int w, int w3)
        throws ApfloatRuntimeException
    {
        DataStorage.Iterator iterator0 = dataStorage0.iterator(DataStorage.READ_WRITE, 0, size),
                             iterator1 = dataStorage1.iterator(DataStorage.READ_WRITE, 0, size),
                             iterator2 = dataStorage2.iterator(DataStorage.READ_WRITE, 0, size);
        int ww = modMultiply(w, w),
                w1 = negate(modDivide((int) 3, (int) 2)),
                w2 = modAdd(w3, modDivide((int) 1, (int) 2)),
                tmp1 = (int) 1,
                tmp2 = (int) 1;

        while (size > 0)
        {
            // 3-point WFTA on the corresponding array elements

            int x0 = iterator0.getInt(),
                    x1 = iterator1.getInt(),
                    x2 = iterator2.getInt(),
                    t;

            if (isInverse)
            {
                // Multiply before transform
                x1 = modMultiply(x1, tmp1);
                x2 = modMultiply(x2, tmp2);
            }

            // Transform columns
            t = modAdd(x1, x2);
            x2 = modSubtract(x1, x2);
            x0 = modAdd(x0, t);
            t = modMultiply(t, w1);
            x2 = modMultiply(x2, w2);
            t = modAdd(t, x0);
            x1 = modAdd(t, x2);
            x2 = modSubtract(t, x2);

            if (!isInverse)
            {
                // Multiply after transform
                x1 = modMultiply(x1, tmp1);
                x2 = modMultiply(x2, tmp2);
            }

            iterator0.setInt(x0);
            iterator1.setInt(x1);
            iterator2.setInt(x2);

            iterator0.next();
            iterator1.next();
            iterator2.next();

            tmp1 = modMultiply(tmp1, w);
            tmp2 = modMultiply(tmp2, ww);

            size--;
        }
    }

    private NTTStrategy factor2Strategy;
}
