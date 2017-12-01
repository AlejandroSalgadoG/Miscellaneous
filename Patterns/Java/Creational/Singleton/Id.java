public class Id{

    private static Id identity = new Id();

    private Id(){}

    public static Id getId(){
        return identity;
    }

    public void consultId(){
        System.out.println("Your id is 1");
    }

}
