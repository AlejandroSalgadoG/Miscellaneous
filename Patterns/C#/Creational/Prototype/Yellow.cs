public class Yellow : Color{

    public Yellow(string info){
        this.info = info;
    }

    public override string getInfo(){
        return info;
    }

    public override Color clone(){
        return new Yellow(info);
    }

}
