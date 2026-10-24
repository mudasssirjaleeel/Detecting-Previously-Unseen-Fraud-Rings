package ec.util;
/** Stand-in for MASON's MersenneTwisterFast: java.util.Random plus the bounded nextLong(n) it adds. */
public class MersenneTwisterFast extends java.util.Random {
    public MersenneTwisterFast(long seed) { super(seed); }
    public long nextLong(long n) {
        if (n <= 0) throw new IllegalArgumentException("n must be positive, got: " + n);
        long bits, val;
        do { bits = (nextLong() << 1) >>> 1; val = bits % n; } while (bits - val + (n - 1) < 0L);
        return val;
    }
}
