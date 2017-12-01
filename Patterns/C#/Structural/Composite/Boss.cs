using System;
using System.Collections.Generic;

public class Boss : Worker{

    private List<Worker> workers;

    public Boss(string name): base(name) {
        workers = new List<Worker>();
    }

    public override void add(Worker worker){
        workers.Add(worker);
    }

    public override void remove(Worker worker){
        workers.Remove(worker);
    }

    public override void displayStructure(){
        Console.WriteLine(name);

        foreach(Worker worker in workers)
            worker.displayStructure();
    }

}
