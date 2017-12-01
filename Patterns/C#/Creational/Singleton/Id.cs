using System;

public class Id{

    private static Id identity = new Id();

    private Id(){}

    public static Id getId(){
        return identity;
    }

    public void consultId(){
        Console.WriteLine("Your id is 1");
    }

}
