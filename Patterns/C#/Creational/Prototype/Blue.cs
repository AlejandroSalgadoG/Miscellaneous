public class Blue : Color{

    public Blue(string info){
        this.info = info;
    }

    public override string getInfo(){
        return info;
    }

    public override Color clone(){
        return new Blue(info);
    }

}
