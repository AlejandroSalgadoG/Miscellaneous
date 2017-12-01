package com.windows;

public class Path {
	static String path = "The path is: \n\nHome \n";
	static public void setPath(String data){ path += data;}
	static public void replacePath(String data){path = path.replace(data, "");}
	static public String getPath(){ return path;}
}
