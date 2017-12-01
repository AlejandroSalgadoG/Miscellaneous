using System;

public class Vest : Coat{

    public override string getName(){
        return "Vest";
    }

    public override void coverChest(){
        Console.WriteLine("Chest covered");
    }

}
