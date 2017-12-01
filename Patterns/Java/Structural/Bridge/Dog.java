public class Dog extends Animal{

    public void jump(){
        movement = new Jump();

        System.out.print("The dog ");
        movement.move();
    }

    public void run(){
        movement = new Run();

        System.out.print("The dog ");
        movement.move();
    }

}
