package org.json;
import java.util.ArrayList;
public class JSONArray { final ArrayList<Object> list = new ArrayList<>();
  public int length(){return list.size();} public Object get(int i){return list.get(i);} }
