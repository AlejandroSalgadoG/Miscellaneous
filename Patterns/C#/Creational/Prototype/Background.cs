using System;

public class Background{

    public void setBackground(Color color){
        Color colorCloned = color.clone();
        Console.WriteLine( "The background is " + colorCloned.getInfo() );
    }

}
