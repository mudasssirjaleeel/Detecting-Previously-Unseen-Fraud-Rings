package sim.engine;
/** Minimal stand-in for MASON's SimState. */
public class SimState implements java.io.Serializable {
    public Schedule schedule = new Schedule();
    public long seed;
    public ec.util.MersenneTwisterFast random;
    public SimState(long seed) { this.seed = seed; this.random = new ec.util.MersenneTwisterFast(seed); }
    public void start() {}
    public void finish() {}
}
