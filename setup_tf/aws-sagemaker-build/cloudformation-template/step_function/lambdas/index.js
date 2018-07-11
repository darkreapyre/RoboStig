var fs=require('fs')
var _=require('lodash')
var lambdas=fs.readdirSync(__dirname).filter(x=>!x.match(/index.js/))

module.exports=Object.assign(
    _.fromPairs(lambdas.map(lambda)),
    {
    "StepLambdaRole":{
      "Type": "AWS::IAM::Role",
      "Properties": {
        "AssumeRolePolicyDocument": {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "Allow",
              "Principal": {
                "Service": "lambda.amazonaws.com"
              },
              "Action": "sts:AssumeRole"
            }
          ]
        },
        "Path": "/",
        "ManagedPolicyArns": [
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AWSCodeBuildAdminAccess",
            "arn:aws:iam::aws:policy/AmazonSNSFullAccess"
        ]
      }
    }
})

function lambda(name){
    var code=fs.readFileSync(__dirname+`/${name}`,'utf-8')
    var js=name.split('.').reverse()[0]==="js"

    return [`StepLambda${name.split('.')[0]}`,{
      "Type": "AWS::Lambda::Function",
      "Properties": {
        "Code": {
            "ZipFile":code
        },
        "Handler":"index.handler",
        "MemorySize": "128",
        "Role": {"Fn::GetAtt": ["StepLambdaRole","Arn"]},
        "Runtime": js ? "nodejs6.10" : "python3.6",
        "Timeout": 60
      }
    }]
}
