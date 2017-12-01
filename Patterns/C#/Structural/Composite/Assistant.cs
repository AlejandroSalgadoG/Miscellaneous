using System;

public class Assistant : Worker{

    public Assistant(string name): base(name) {}

    public override void displayStructure(){
        Console.WriteLine(name);
    }

    public override void add(Worker worker){}
    public override void remove(Worker worker){}

}
