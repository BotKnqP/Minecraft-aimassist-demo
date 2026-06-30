package com.mcbowagent.record;

import java.util.Locale;

/**
 * Tiny hand-rolled JSON helpers so the on-disk field names EXACTLY match
 * python/mc_bow_agent/data_schema.py (snake_case), with no reflection surprises.
 */
public final class JsonUtil {

    private JsonUtil() {}

    /** Fixed-precision number, Locale.US (so '.' decimal separator, never ','). */
    public static String num(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return "0";
        return String.format(Locale.US, "%.4f", v);
    }

    public static String str(String s) {
        if (s == null) return "null";
        StringBuilder b = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default: b.append(c);
            }
        }
        return b.append("\"").toString();
    }

    public static String bool(boolean v) {
        return v ? "true" : "false";
    }

    public static String vec3(double x, double y, double z) {
        return "[" + num(x) + "," + num(y) + "," + num(z) + "]";
    }
}
