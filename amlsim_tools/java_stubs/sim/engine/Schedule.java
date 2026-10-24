package sim.engine;
import java.util.ArrayList;
/** Minimal stand-in for MASON's Schedule: repeating agents stepped once per step, in insertion order. */
public class Schedule implements java.io.Serializable {
    private final ArrayList<Steppable> agents = new ArrayList<>();
    private long steps = 0;
    public long getSteps() { return steps; }
    public void scheduleRepeating(Steppable s) { agents.add(s); }
    public boolean step(SimState state) {
        for (Steppable s : agents) s.step(state);
        steps++;
        return true;
    }
}
