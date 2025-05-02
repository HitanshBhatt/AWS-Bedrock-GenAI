from aws_cdk import (
    aws_s3,
    Stack,
    Duration,
    CfnOutput   #Will write the generaed S3 bucket name to the CloudFormation stack output
)
from constructs import Construct

class PyStarterStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.bucket = aws_s3.Bucket(self, "PyBucket",   #PyBucket is CDK ID for the S3 bucket
            lifecycle_rules=[   #LifeCycle rules is a list of rules that define the lifecycle of the S3 bucket
                aws_s3.LifecycleRule(
                    expiration=Duration.days(3)
                )
            ]                        
        )

        #Output the bucket name to the CloudFormation stack output
        CfnOutput(self, "PyBucketName",
                value=self.bucket.bucket_name
                )
