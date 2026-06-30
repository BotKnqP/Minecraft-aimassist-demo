package com.mcbowagent.state;

import com.mcbowagent.record.JsonUtil;

/** Per-tick combat events. v0 fills arrow_released + damage_taken reliably;
 *  arrow_hit / kill are best-effort (see TickRecorder TODOs). */
public final class EventSnapshot {

    public final boolean arrowReleased;
    public final boolean arrowHit;
    public final boolean kill;
    public final boolean damageTaken;

    public EventSnapshot(boolean arrowReleased, boolean arrowHit, boolean kill, boolean damageTaken) {
        this.arrowReleased = arrowReleased;
        this.arrowHit = arrowHit;
        this.kill = kill;
        this.damageTaken = damageTaken;
    }

    public void writeObject(StringBuilder sb) {
        sb.append('{');
        sb.append("\"arrow_released\":").append(JsonUtil.bool(arrowReleased)).append(',');
        sb.append("\"arrow_hit\":").append(JsonUtil.bool(arrowHit)).append(',');
        sb.append("\"kill\":").append(JsonUtil.bool(kill)).append(',');
        sb.append("\"damage_taken\":").append(JsonUtil.bool(damageTaken));
        sb.append('}');
    }
}
