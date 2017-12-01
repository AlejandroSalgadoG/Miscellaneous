using System;

public class Beret : Hat{

    public override string getName(){
        return "Beret";
    }

    public override void coverHead(){
        Console.WriteLine("Head covered");
    }

}
