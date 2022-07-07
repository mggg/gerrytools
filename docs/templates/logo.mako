
<!-- include a script for adding stuff to the end of proofs. -->
<script>
    // Get all the proofs in the document.
    proofs = document.getElementsByClassName("proof");

    // For each of the proofs, attach a floating child element in the bottom-right
    // corner.
    for (var proof of proofs) {
        // Create a proof-ending tombstone.
        square = document.createElement("div");
        square.className = "tombstone";
        square.innerHTML = "◼️";

        // Attach the tombstone to the proof.
        proof.appendChild(square);
    }
</script>

<header>
    <a class="homelink" rel="home" title="gerrytools" href="https://github.com/mggg/gerrytools">
        <style>
            header > h1 { display: none; }

            img.resize {
                max-width: 80%;
                max-height: 80%;
                display: block;
                margin: 0 auto;
            }

            div.proof {
                border: 1px solid black;
                padding: 0em 1em;
                width: 90%;
                margin: 1em auto;
            }

            .tombstone {
                margin-top: -2em;
                float: right;
            }
        </style>
        <img class="resize" src="https://mggg.org/assets/logo.svg" alt="MGGG Logo">
    </a>
</header>