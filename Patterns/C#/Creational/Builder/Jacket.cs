using System;

public class Jacket : Coat{

    public override string getName(){
        return "Jacket";
    }

    public override void coverChest(){
        Console.WriteLine("Chest covered");
    }

}
