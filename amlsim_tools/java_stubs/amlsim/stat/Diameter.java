package amlsim.stat;
/** Stand-in: AMLSim's optional diameter statistic needs WebGraph; it is disabled (compute_diameter=false). */
public class Diameter {
    public Diameter(int numAccounts) {}
    public void addEdge(String srcID, String dstID) {}
    public double[] computeDiameter() { return new double[]{0, 0}; }
}
