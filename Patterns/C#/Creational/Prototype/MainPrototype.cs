public class MainPrototype{

    public static void Main(string[] args){

        Background background = new Background();

        Color yellow = new Yellow("color yellow");
        Color blue = new Blue("color blue");
        Color red = new Red("color red");

        background.setBackground(yellow);
        background.setBackground(blue);
        background.setBackground(red);
    }

}
