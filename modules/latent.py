import torch

def get_encoded_latents(vae, testloader, device, pos = 'pre'):
    vae.eval()  # set vae to evaluation mode
    all_latents = []

    with torch.no_grad():
        for batch_idx, (data, _, _) in enumerate(testloader):
            data = data.to(device)
            pre_mu, pre_logvar, post_mu, post_logvar = vae.encode(data)
            if pos == 'pre':
                z = vae.reparameterize(pre_mu, pre_logvar)
            elif pos == 'post':
                z = vae.reparameterize(post_mu, post_logvar)
            else:
                raise ValueError('pos type is not defined') 
            all_latents.append(z.cpu())

    # Stack all the latents together
    all_latents = torch.cat(all_latents, dim=0)

    return all_latents