using System;

public class Cap : Hat{

    public override string getName(){
        return "Cap";
    }

    public override void coverHead(){
        Console.WriteLine("Head covered");
    }

}
