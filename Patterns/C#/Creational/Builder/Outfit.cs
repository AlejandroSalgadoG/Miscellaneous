using System;
using System.Collections.Generic;

public class Outfit{

    private List<Clothe> shoped;

    public Outfit(){
        shoped = new List<Clothe>();
    }

    public void addClothe(Clothe clothe){
        shoped.Add(clothe);
    }

    public void listClothe(){
        Console.WriteLine("Outfit: ");
        foreach(Clothe clothe in shoped)
            Console.WriteLine("\t" + clothe.getName() );
    }

}
