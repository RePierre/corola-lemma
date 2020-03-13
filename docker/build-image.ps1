param(
    [parameter(Mandatory=$true)]
    [string]$SshKeysDirectory,
    [parameter(Mandatory=$true)]
    [string]$Password
)

$privateKey = Get-Content -Path ([System.IO.Path]::Combine($SshKeysDirectory, "id_rsa"))
$publicKey = Get-Content -Path ([System.IO.Path]::Combine($SshKeysDirectory, "id_rsa.pub"))

docker build -t corola-lemma:development --build-arg ssh_prv_key="$privateKey" --build-arg ssh_pub_key="$publicKey" --build-arg password=$Password .
