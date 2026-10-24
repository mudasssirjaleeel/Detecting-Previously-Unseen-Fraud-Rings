package org.json;
import java.util.*;
/** Minimal JSON object/parser with the subset of the org.json API that AMLSim uses. */
public class JSONObject {
    private final Map<String,Object> map = new LinkedHashMap<>();
    public JSONObject() {}
    public JSONObject(String s) { P p = new P(s); p.ws(); JSONObject o = p.obj(); map.putAll(o.map); }
    public boolean has(String k){ return map.containsKey(k); }
    public Set<String> keySet(){ return map.keySet(); }
    public Iterator<String> keys(){ return map.keySet().iterator(); }
    public Object get(String k){ if(!map.containsKey(k)) throw new JSONException("JSONObject[\""+k+"\"] not found."); return map.get(k); }
    public Object opt(String k){ return map.get(k); }
    public JSONObject getJSONObject(String k){ return (JSONObject) get(k); }
    public JSONArray getJSONArray(String k){ return (JSONArray) get(k); }
    public String getString(String k){ return String.valueOf(get(k)); }
    public String optString(String k, String d){ Object v=map.get(k); return v==null?d:String.valueOf(v); }
    private static double num(Object v){ if(v instanceof Number) return ((Number)v).doubleValue(); return Double.parseDouble(String.valueOf(v)); }
    public double getDouble(String k){ return num(get(k)); }
    public float getFloat(String k){ return (float) num(get(k)); }
    public int getInt(String k){ return (int) num(get(k)); }
    public long getLong(String k){ return (long) num(get(k)); }
    public int optInt(String k, int d){ return map.containsKey(k)?getInt(k):d; }
    public double optDouble(String k, double d){ return map.containsKey(k)?getDouble(k):d; }
    public boolean getBoolean(String k){ Object v=get(k); if(v instanceof Boolean) return (Boolean)v; return Boolean.parseBoolean(String.valueOf(v)); }
    public boolean optBoolean(String k, boolean d){ return map.containsKey(k)?getBoolean(k):d; }
    public String toString(){ return map.toString(); }

    private static final class P {
        final String s; int i = 0; P(String s){ this.s = s; }
        void ws(){ while(i<s.length()){ char c=s.charAt(i);
            if(Character.isWhitespace(c)) i++;
            else if(c=='/' && i+1<s.length() && s.charAt(i+1)=='/'){ while(i<s.length() && s.charAt(i)!='\n') i++; }
            else break; } }
        JSONObject obj(){ JSONObject o = new JSONObject(); expect('{'); ws();
            if(peek()=='}'){ i++; return o; }
            while(true){ ws(); String k = str(); ws(); expect(':'); ws(); o.map.put(k, val()); ws();
                char c = s.charAt(i++); if(c=='}') return o; if(c!=',') throw new JSONException("bad object at "+i); ws();
                if(peek()=='}'){ i++; return o; } } }
        JSONArray arr(){ JSONArray a = new JSONArray(); expect('['); ws();
            if(peek()==']'){ i++; return a; }
            while(true){ ws(); a.list.add(val()); ws(); char c = s.charAt(i++); if(c==']') return a; if(c!=',') throw new JSONException("bad array at "+i); } }
        Object val(){ char c = peek();
            if(c=='{') return obj(); if(c=='[') return arr(); if(c=='"') return str();
            if(s.startsWith("true",i)){ i+=4; return Boolean.TRUE; } if(s.startsWith("false",i)){ i+=5; return Boolean.FALSE; }
            if(s.startsWith("null",i)){ i+=4; return null; }
            int st=i; while(i<s.length() && "+-0123456789.eE".indexOf(s.charAt(i))>=0) i++;
            String n = s.substring(st,i); if(n.isEmpty()) throw new JSONException("bad value at "+i);
            if(n.contains(".")||n.contains("e")||n.contains("E")) return Double.parseDouble(n);
            long L = Long.parseLong(n); return (L==(int)L)?(Object)Integer.valueOf((int)L):(Object)Long.valueOf(L); }
        String str(){ expect('"'); StringBuilder b = new StringBuilder();
            while(true){ char c = s.charAt(i++); if(c=='"') return b.toString();
                if(c=='\\'){ char e = s.charAt(i++); switch(e){ case 'n': b.append('\n'); break; case 't': b.append('\t'); break;
                    case 'u': b.append((char)Integer.parseInt(s.substring(i,i+4),16)); i+=4; break; default: b.append(e);} }
                else b.append(c); } }
        char peek(){ return s.charAt(i); }
        void expect(char c){ if(s.charAt(i)!=c) throw new JSONException("expected "+c+" at "+i+" got "+s.charAt(i)); i++; }
    }
}
