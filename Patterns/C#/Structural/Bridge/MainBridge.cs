public class MainBridge{

    public static void Main(string[] args){

        Animal dog = new Dog();
        Animal cat = new Cat();

        dog.jump();
        dog.run();

        cat.jump();
        cat.run();

    }

}
