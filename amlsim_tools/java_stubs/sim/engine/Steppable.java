package sim.engine;
/** Minimal stand-in for MASON's Steppable (only the API AMLSim uses). */
public interface Steppable extends java.io.Serializable { void step(SimState state); }
